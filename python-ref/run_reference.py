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

REPO_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def build_opf_config(hf_config: dict) -> dict:
    """Translate transformers-style config.json into opf's artifact-contract schema."""
    rope = hf_config["rope_parameters"]
    sliding_half = int(hf_config["sliding_window"])
    bandwidth = sliding_half * 2 + 1
    return {
        # opf REQUIRED_ENCODER_CONFIG_KEYS
        "model_type": "privacy_filter",
        "encoding": hf_config["encoding"],
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


def materialize_opf_checkpoint(src_models_dir: Path, dst_dir: Path) -> Path:
    """Build a side-car checkpoint dir that opf can load.

    The dir contains a translated config.json plus a symlink to the original
    safetensors weights. The original directory is left untouched.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_config = json.loads((src_models_dir / "config.json").read_text())
    opf_config = build_opf_config(src_config)
    (dst_dir / "config.json").write_text(json.dumps(opf_config, indent=2))

    src_weights = (src_models_dir / "model.safetensors").resolve()
    link = dst_dir / "model.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(src_weights)
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
) -> Reference:
    # Defer the import so --help works even if opf isn't installed.
    from opf import OPF, DecodeOptions  # noqa: F401

    workdir = REPO_ROOT / "python-ref" / ".opf-checkpoint"
    materialize_opf_checkpoint(models_dir, workdir)

    opf_obj = OPF(
        model=str(workdir),
        device="cpu",
        decode_mode=decoder,
        context_window_length=context_window_length,
        trim_whitespace=True,
        discard_overlapping_predicted_spans=False,
    )
    if decoder == "viterbi" and calibration_path is not None:
        opf_obj.set_viterbi_decoder(calibration_path=str(calibration_path))

    runtime, dec = opf_obj.get_prediction_components()
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
    p.add_argument("-m", "--model-dir", type=Path, default=REPO_ROOT / "models")
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
    args = p.parse_args()

    log(f"loading opf reference (decoder={args.decoder}, n_ctx={args.context_window_length})...")
    ref = load_reference(
        models_dir=args.model_dir,
        decoder=args.decoder,
        context_window_length=args.context_window_length,
        calibration_path=args.calibration,
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
