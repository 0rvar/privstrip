#!/usr/bin/env python3
"""
JSONL-stream wrapper around the official openai/privacy-filter `opf` package.

The HF checkpoint at openai/privacy-filter and the `opf` GitHub package were
released a few days apart with mismatched config schemas. The HF config.json
uses transformers-style field names (num_local_experts, rope_parameters.*,
sliding_window=128 as a half-width, etc.) and is missing keys that opf's
artifact-contract validator requires (model_type="privacy_filter",
bidirectional_context, bidirectional_left_context, bidirectional_right_context,
num_labels, param_dtype, flat rope_*).

This wrapper rewrites the HF config into the schema opf expects, materializes
it in a side-car directory, and points opf at that directory while the actual
weights file is symlinked from the source models/ dir. Nothing about the
weights themselves is touched.

Stream protocol (matches scripts/validate.ts):
    in:  {"id": <any>, "text": "..."} per line on stdin
    out: {"id": <same>, "spans": [{"label","byte_start","byte_end","text"}],
          "tokens": <int>, "elapsed_us": <int>} per line on stdout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def build_opf_config(hf_config: dict) -> dict:
    """Translate transformers-style config.json into opf's artifact-contract schema."""
    rope = hf_config["rope_parameters"]
    sliding_half = int(hf_config["sliding_window"])
    bandwidth = sliding_half * 2 + 1
    # OpenMed/privacy-filter-multilingual exposes the tokenizer name under
    # opf_metadata.encoding. Upstream openai/privacy-filter's config.json
    # carries neither field, so default to o200k_base — the tokenizer it
    # ships with tokenizer.json.
    opf_metadata = hf_config.get("opf_metadata") or {}
    encoding = hf_config.get("encoding") or opf_metadata.get("encoding") or "o200k_base"

    # OPF resolves the label space either from a known `category_version`
    # (33/57/101 classes) or from explicit top-level `span_class_names` /
    # `ner_class_names`. The multilingual checkpoint's 217 classes are not a
    # built-in version, so we promote the names from `opf_metadata` to top
    # level to trigger OPF's custom-label-space path.
    custom_span_class_names = opf_metadata.get("span_class_names")
    custom_ner_class_names = opf_metadata.get("ner_class_names")
    custom_category_version = opf_metadata.get("category_version")

    out: dict = {
        # opf REQUIRED_ENCODER_CONFIG_KEYS
        "model_type": "privacy_filter",
        "encoding": encoding,
        "num_hidden_layers": hf_config["num_hidden_layers"],
        "num_experts": hf_config["num_local_experts"],
        "experts_per_token": hf_config["num_experts_per_tok"],
        "vocab_size": hf_config["vocab_size"],
        "num_labels": len(hf_config["id2label"]),
        "hidden_size": hf_config["hidden_size"],
        "intermediate_size": hf_config["intermediate_size"],
        "head_dim": hf_config["head_dim"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "num_key_value_heads": hf_config["num_key_value_heads"],
        # In opf's bidirectional path, sliding_window is the full bandwidth
        # (left+right+1), not the half-width that the HF config stores.
        "sliding_window": bandwidth,
        "bidirectional_context": True,
        "bidirectional_left_context": sliding_half,
        "bidirectional_right_context": sliding_half,
        "initial_context_length": int(hf_config["initial_context_length"]),
        "rope_theta": float(rope["rope_theta"]),
        "rope_scaling_factor": float(rope["factor"]),
        # HF's beta_slow corresponds to opf's ntk_alpha; HF's beta_fast to ntk_beta.
        # The Rust port already encodes this mapping in build_yarn_rope; preserve it.
        "rope_ntk_alpha": float(rope["beta_slow"]),
        "rope_ntk_beta": float(rope["beta_fast"]),
        "param_dtype": hf_config.get("dtype", "bfloat16"),
        # Keep label space metadata so opf's label-space resolver finds them.
        "id2label": hf_config["id2label"],
        "label2id": hf_config["label2id"],
        # opf's label-space resolver reads ner_class_names from id2label/label2id; nothing else needed.
        # Carry default_n_ctx forward; we still pass an explicit context_window_length below.
        "default_n_ctx": int(hf_config.get("default_n_ctx", 8192)),
        "swiglu_limit": float(hf_config.get("swiglu_limit", 7.0)),
    }
    if custom_span_class_names is not None:
        out["span_class_names"] = list(custom_span_class_names)
    if custom_ner_class_names is not None:
        out["ner_class_names"] = list(custom_ner_class_names)
    if custom_category_version is not None:
        out["category_version"] = str(custom_category_version)
    return out


def _is_hf_naming(weights_path: Path) -> bool:
    with safe_open(str(weights_path), framework="pt") as f:
        return "model.embed_tokens.weight" in set(f.keys())


def _convert_hf_to_opf_weights(
    src: Path,
    dst_weights: Path,
    classifier_bias_dst: Path,
    num_layers: int,
) -> bool:
    """Rewrite an HF-naming safetensors as an OPF-naming safetensors.

    Returns True if the source had a classifier-head bias (`score.bias`), which
    OPF's bias-free unembedding cannot honor — load_reference adds that bias
    back via a forward hook so logits match the rust port byte-for-byte.

    Tensor mappings (HF → OPF):
      model.embed_tokens.weight                              → embedding.weight
      model.norm.weight                                      → norm.scale
      score.weight                                           → unembedding.weight
      score.bias                                             → (sidecar) unembedding.bias
      model.layers.{i}.input_layernorm.weight                → block.{i}.attn.norm.scale
      model.layers.{i}.post_attention_layernorm.weight       → block.{i}.mlp.norm.scale
      model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}  → cat → block.{i}.attn.qkv.{weight,bias}
      model.layers.{i}.self_attn.o_proj.{weight,bias}        → block.{i}.attn.out.{weight,bias}
      model.layers.{i}.self_attn.sinks                       → block.{i}.attn.sinks
      model.layers.{i}.mlp.router.{weight,bias}              → block.{i}.mlp.gate.{weight,bias}
      model.layers.{i}.mlp.experts.gate_up_proj{,_bias}      → block.{i}.mlp.swiglu.{weight,bias}
      model.layers.{i}.mlp.experts.down_proj{,_bias}         → block.{i}.mlp.out.{weight,bias}
    """
    tensors: dict[str, torch.Tensor] = {}
    classifier_bias: torch.Tensor | None = None
    with safe_open(str(src), framework="pt") as f:
        keys = set(f.keys())
        tensors["embedding.weight"] = f.get_tensor("model.embed_tokens.weight")
        tensors["norm.scale"] = f.get_tensor("model.norm.weight")
        tensors["unembedding.weight"] = f.get_tensor("score.weight")
        if "score.bias" in keys:
            classifier_bias = f.get_tensor("score.bias")

        for i in range(num_layers):
            sl = f"model.layers.{i}"
            bl = f"block.{i}"
            tensors[f"{bl}.attn.norm.scale"] = f.get_tensor(f"{sl}.input_layernorm.weight")
            tensors[f"{bl}.mlp.norm.scale"] = f.get_tensor(f"{sl}.post_attention_layernorm.weight")
            qw = f.get_tensor(f"{sl}.self_attn.q_proj.weight")
            kw = f.get_tensor(f"{sl}.self_attn.k_proj.weight")
            vw = f.get_tensor(f"{sl}.self_attn.v_proj.weight")
            tensors[f"{bl}.attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0).contiguous()
            qb = f.get_tensor(f"{sl}.self_attn.q_proj.bias")
            kb = f.get_tensor(f"{sl}.self_attn.k_proj.bias")
            vb = f.get_tensor(f"{sl}.self_attn.v_proj.bias")
            tensors[f"{bl}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0).contiguous()
            tensors[f"{bl}.attn.out.weight"] = f.get_tensor(f"{sl}.self_attn.o_proj.weight")
            tensors[f"{bl}.attn.out.bias"] = f.get_tensor(f"{sl}.self_attn.o_proj.bias")
            tensors[f"{bl}.attn.sinks"] = f.get_tensor(f"{sl}.self_attn.sinks")
            tensors[f"{bl}.mlp.gate.weight"] = f.get_tensor(f"{sl}.mlp.router.weight")
            tensors[f"{bl}.mlp.gate.bias"] = f.get_tensor(f"{sl}.mlp.router.bias")
            tensors[f"{bl}.mlp.swiglu.weight"] = f.get_tensor(f"{sl}.mlp.experts.gate_up_proj")
            tensors[f"{bl}.mlp.swiglu.bias"] = f.get_tensor(f"{sl}.mlp.experts.gate_up_proj_bias")
            tensors[f"{bl}.mlp.out.weight"] = f.get_tensor(f"{sl}.mlp.experts.down_proj")
            tensors[f"{bl}.mlp.out.bias"] = f.get_tensor(f"{sl}.mlp.experts.down_proj_bias")

    save_file(tensors, str(dst_weights))
    if classifier_bias is not None:
        save_file({"unembedding.bias": classifier_bias}, str(classifier_bias_dst))
        return True
    if classifier_bias_dst.exists():
        classifier_bias_dst.unlink()
    return False


def materialize_opf_checkpoint(src_models_dir: Path, dst_dir: Path) -> Path:
    """Build a side-car checkpoint dir that opf can load.

    For an upstream-naming checkpoint (openai/privacy-filter), the dir contains
    a translated config.json plus a symlink to the original safetensors weights.

    For an HF-naming checkpoint (e.g. OpenMed/privacy-filter-multilingual), the
    weights are rewritten into OPF's expected naming, and any classifier-head
    bias is split out into a sidecar that load_reference applies via a forward
    hook (OPF's unembedding is `bias=False` and silently drops it otherwise).

    Conversion is cached on src mtime so swapping models doesn't pay the rewrite
    cost on every load. The original models/ directory is left untouched.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_config = json.loads((src_models_dir / "config.json").read_text())
    opf_config = build_opf_config(src_config)
    (dst_dir / "config.json").write_text(json.dumps(opf_config, indent=2))

    src_weights = (src_models_dir / "model.safetensors").resolve()
    dst_weights = dst_dir / "model.safetensors"
    classifier_bias_path = dst_dir / "classifier_bias.safetensors"

    if _is_hf_naming(src_weights):
        # Rewrite once; reuse on subsequent loads. Stale check uses src mtime.
        needs_rewrite = (
            not dst_weights.exists()
            or dst_weights.is_symlink()  # leftover from a base-model load — replace with a real file
            or src_weights.stat().st_mtime > dst_weights.stat().st_mtime
        )
        if needs_rewrite:
            log(f"converting HF-naming weights → OPF naming at {dst_weights}...")
            if dst_weights.is_symlink() or dst_weights.exists():
                dst_weights.unlink()
            _convert_hf_to_opf_weights(
                src_weights,
                dst_weights,
                classifier_bias_path,
                num_layers=int(src_config["num_hidden_layers"]),
            )
    else:
        if dst_weights.is_symlink() or dst_weights.exists():
            dst_weights.unlink()
        dst_weights.symlink_to(src_weights)
        # If we previously converted an HF checkpoint into this workdir, drop
        # the stale classifier bias so it doesn't leak into a base-model load.
        if classifier_bias_path.exists():
            classifier_bias_path.unlink()
    return dst_dir


@dataclass
class Reference:
    opf_obj: object  # opf.OPF instance
    runtime: object  # opf InferenceRuntime
    decoder: object  # ViterbiCRFDecoder or None for argmax
    decode_mode: str
    label_info: object


def load_reference(
    models_dir: Path,
    decoder: str,
    context_window_length: int,
    calibration_path: Path | None,
    device: str,
) -> Reference:
    # Defer the import so --help works even if opf isn't installed.
    from opf import OPF, DecodeOptions  # noqa: F401

    workdir = REPO_ROOT / "python-ref" / f".opf-checkpoint-{models_dir.name}"
    materialize_opf_checkpoint(models_dir, workdir)

    opf_obj = OPF(
        model=str(workdir),
        device=device,
        decode_mode=decoder,
        context_window_length=context_window_length,
        trim_whitespace=True,
        discard_overlapping_predicted_spans=False,
    )
    if decoder == "viterbi" and calibration_path is not None:
        opf_obj.set_viterbi_decoder(calibration_path=str(calibration_path))

    runtime, dec = opf_obj.get_prediction_components()

    # OPF's classifier head is bias=False (opf/_model/model.py: `self.unembedding
    # = nn.Linear(..., bias=False)` and `F.linear(x, self.unembedding.weight,
    # None)`). HF-naming checkpoints (multilingual etc.) carry a `score.bias`
    # that we extracted into a sidecar at materialize time; add it back via a
    # forward hook so the python reference matches the rust port (which already
    # honors the bias).
    classifier_bias_path = workdir / "classifier_bias.safetensors"
    if classifier_bias_path.exists():
        with safe_open(str(classifier_bias_path), framework="pt") as f:
            bias = f.get_tensor("unembedding.bias")
        target_device = next(runtime.model.parameters()).device
        target_dtype = next(runtime.model.parameters()).dtype
        bias = bias.to(device=target_device, dtype=target_dtype)
        runtime.model.register_forward_hook(
            lambda _module, _inputs, output: output + bias
        )

    return Reference(
        opf_obj=opf_obj,
        runtime=runtime,
        decoder=dec,
        decode_mode=decoder,
        label_info=runtime.label_info,
    )


def char_index_to_byte(text: str, char_idx: int) -> int:
    return len(text[:char_idx].encode("utf-8"))


def run_predict(ref: Reference, text: str) -> dict:
    """Run one prediction and return the JSONL response payload (without id)."""
    if not text:
        return {"spans": [], "tokens": 0, "elapsed_us": 0}

    started = time.perf_counter_ns()
    result = ref.opf_obj.redact(text)
    elapsed_us = (time.perf_counter_ns() - started) // 1000

    # OPF returns char offsets into result.text. If the tokenizer round-trip
    # produces a different string (decoded_mismatch=True), result.text is the
    # decoded form, which may not match our input byte-for-byte. Convert to
    # byte offsets in result.text either way; the comparator compares spans by
    # (label, byte_start, byte_end, snippet), so they need to refer to the same
    # source string. When there's a mismatch, surface a warning so callers can
    # ignore that row in the agreement matrix.
    source_text = result.text  # decoded round-trip form, equals input when no mismatch
    spans_out = []
    for span in result.detected_spans:
        bs = char_index_to_byte(source_text, int(span.start))
        be = char_index_to_byte(source_text, int(span.end))
        spans_out.append({
            "label": span.label,
            "byte_start": bs,
            "byte_end": be,
            "text": span.text,
        })

    payload = {"spans": spans_out, "elapsed_us": int(elapsed_us)}
    if result.warning is not None:
        payload["decoded_mismatch"] = True
    # Token count: opf doesn't return it; derive from the encoding for parity with Rust.
    token_ids = ref.runtime.encoding.encode(text, allowed_special="all")
    payload["tokens"] = len(token_ids)
    return payload


def run_stream(ref: Reference) -> int:
    out = sys.stdout
    for line in sys.stdin:
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            out.write(json.dumps({"id": None, "error": f"invalid input json: {exc}"}) + "\n")
            out.flush()
            continue
        rid = req.get("id")
        text = req.get("text", "")
        try:
            resp = run_predict(ref, text)
        except Exception as exc:  # noqa: BLE001 - boundary between subprocess and harness
            out.write(json.dumps({"id": rid, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            out.flush()
            continue
        resp["id"] = rid
        out.write(json.dumps(resp) + "\n")
        out.flush()
    return 0


def run_debug(ref: Reference, text: str) -> int:
    """Emit per-token (token_id, char range, decoded text, label, argmax_prob)."""
    if not text:
        return 0
    runtime = ref.runtime
    token_ids = runtime.encoding.encode(text, allowed_special="all")
    if not token_ids:
        return 0
    window_tokens = torch.tensor([token_ids], device=runtime.device, dtype=torch.int32)
    attention_mask = torch.ones_like(window_tokens, dtype=torch.bool)
    with torch.inference_mode():
        logits = runtime.model(window_tokens, attention_mask=attention_mask)
        log_probs = F.log_softmax(logits.float(), dim=-1)[0].cpu()
    span_class_names = runtime.label_info.span_class_names
    # opf encodes labels as ints from build_label_info; expose the BIES string via id2label round-trip.
    # Easiest: rebuild by reading config id2label.
    config_path = Path(runtime.checkpoint) / "config.json"
    cfg = json.loads(config_path.read_text())
    id2label = {int(k): v for k, v in cfg["id2label"].items()}

    if ref.decoder is not None:
        decoded_raw = ref.decoder.decode(log_probs)
        decoded = decoded_raw.tolist() if hasattr(decoded_raw, "tolist") else list(decoded_raw)
    else:
        decoded = log_probs.argmax(dim=1).tolist()

    dump_top_k = int(os.environ.get("PRIVSTRIP_DEBUG_TOPK", "0"))

    # Reconstruct char offsets per token via single-token decode.
    cursor = 0
    for i, tid in enumerate(token_ids):
        try:
            piece = runtime.encoding.decode_single_token_bytes(tid).decode("utf-8", errors="replace")
        except Exception:
            piece = "?"
        char_len = len(piece)
        char_start = cursor
        char_end = cursor + char_len
        cursor = char_end
        token_lp = log_probs[i]
        argmax_id = int(token_lp.argmax().item())
        argmax_prob = float(token_lp.max().exp().item())
        decoded_id = int(decoded[i])
        sys.stdout.write(
            f"{i:>3} tok={tid:<7} chars={char_start}..{char_end} "
            f"text={piece!r} {ref.decode_mode}={id2label.get(decoded_id, '?')}({decoded_id}) "
            f"argmax={id2label.get(argmax_id, '?')}({argmax_prob:.2f})\n"
        )
        if dump_top_k > 0:
            top_vals, top_ids = torch.topk(token_lp, k=dump_top_k)
            parts = [
                f"{id2label.get(int(idx), '?')}={float(val):+.6f}"
                for idx, val in zip(top_ids.tolist(), top_vals.tolist())
            ]
            sys.stdout.write(f"    top: {' '.join(parts)}\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["stream", "debug"], default="stream", nargs="?")
    p.add_argument("-m", "--model-dir", type=Path, default=REPO_ROOT / "models/base")
    p.add_argument("--decoder", choices=["viterbi", "argmax"], default="viterbi")
    p.add_argument(
        "--context-window-length",
        type=int,
        default=8192,
        help="Override opf's CPU default of 4096; matches the Rust RoPE table cap.",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Optional path to viterbi_calibration.json (default: ignored)",
    )
    p.add_argument("-t", "--text", type=str, default=None, help="For debug mode")
    p.add_argument("-f", "--file", type=Path, default=None, help="For debug mode")
    p.add_argument(
        "--mps",
        action="store_true",
        help="Run on Apple GPU via PyTorch's MPS backend instead of CPU. "
             "Performance only — validation should stay on CPU for determinism.",
    )
    args = p.parse_args()

    device = "mps" if args.mps else "cpu"
    if args.mps:
        # opf forces Triton-backed MoE kernels when device != cpu, but Triton
        # has no Apple-Silicon backend. Setting OPF_MOE_TRITON=0 makes opf
        # fall back to the plain PyTorch MoE path on MPS.
        os.environ.setdefault("OPF_MOE_TRITON", "0")
    log(
        f"loading opf reference (decoder={args.decoder}, "
        f"n_ctx={args.context_window_length}, device={device})..."
    )
    ref = load_reference(
        models_dir=args.model_dir,
        decoder=args.decoder,
        context_window_length=args.context_window_length,
        calibration_path=args.calibration,
        device=device,
    )
    log("ready")

    if args.mode == "stream":
        return run_stream(ref)

    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = args.file.read_text()
    else:
        text = sys.stdin.read()
    return run_debug(ref, text)


if __name__ == "__main__":
    sys.exit(main())
