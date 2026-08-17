# privstrip

A Rust **library and CLI** for detecting personally identifiable information (PII) in text using the [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model. Runs the model directly via candle + safetensors — no Python, no ONNX runtime.

The library is the inference engine and nothing else. The HTTP service that wraps it for deployment lives in the Timely infra repo at `services/privstrip` — see "HTTP service" below.

## What it is

`privstrip` ingests text (file, string, or stdin) and emits one of:

- `check` — print PII locations to stderr; exit 1 if any are found, 0 otherwise
- `redact` — print input with PII replaced by `<LABEL>` placeholders
- `list` — print a JSON list of detected spans
- `debug` — per-token predictions for debugging the decoder
- `stream` — read JSONL `{"id":..., "text":...}` from stdin, emit `{"id":..., "spans":[...]}` per line

The model recognizes 8 entity types (`account_number`, `private_address`, `private_date`, `private_email`, `private_person`, `private_phone`, `private_url`, `secret`) using BIES tagging — 33 classes total (8 × 4 + `O`).

## Layout

- `src/` — Rust source
  - `lib.rs` — the library surface: what `services/privstrip` in the infra repo consumes
  - `engine.rs` — `Engine`: tokenize → forward → decode → extract spans, plus the shared JSON envelope
  - `main.rs` — CLI only: arg parsing, I/O, run modes. Consumes the library like any other dependent.
  - `model.rs` — custom transformer architecture (GQA + sparse MoE + YaRN RoPE + bidirectional sliding-window attention)
  - `viterbi.rs` — constraint-aware BIES decoder
  - `labels.rs` — config-driven label/boundary metadata
  - `spans.rs` — token-id → byte-span extraction, whitespace trimming, dedup
  - `config.rs` — model-config deserialization
  - `timing.rs` — opt-in per-stage wall-clock instrumentation gated by `PRIVSTRIP_TIMING=1`
- `models/` — model artifacts (gitignored). One subdirectory per checkpoint. Each subdir contains:
    - `model.safetensors` (~2.8 GB bf16)
    - `tokenizer.json`, `tokenizer_config.json` — o200k_base BPE (identical across checkpoints in this family)
    - `config.json` — architecture hyperparameters and `id2label`
    - `viterbi_calibration.json` — Viterbi transition biases. Auto-loaded at startup; the shipped `default` operating point has all-zero biases (a no-op vs constraint-only decoding). Override with `--operating-point <name>`.
  - `models/base/` — `openai/privacy-filter` upstream weights. Custom on-disk tensor naming (`block.{i}.attn.qkv.weight` etc.). 33 BIES classes for 8 PII categories. **The default `--model-dir` and the only checkpoint validated against the OPF Python reference (C).** Also contains `onnx/` — ONNX export from upstream (kept for the transformers.js oracle, see below). Candle cannot run the ONNX (see "What didn't work").
  - `models/multilingual/` — `OpenMed/privacy-filter-multilingual`. Same architecture but **HF-standard tensor naming** (`model.layers.{i}.self_attn.q_proj.weight` etc., separate Q/K/V, `score.{weight,bias}` classifier head). 217 BIES classes for 54 PII categories across 16 languages. The Rust loader detects naming via `vb.contains_tensor("embedding.weight")` and dispatches accordingly — both layouts share the same forward pass.
- `python-ref/` — Python reference (oracle C). Standalone uv project that wraps the official `opf` package and exposes the same JSONL stream protocol as `privstrip stream`. See `python-ref/README.md`.
- `flake.nix` — dev shell providing cargo + rustc + uv + python311 + bun + samply.
- `scripts/`
  - `validate.ts` — primary harness. Spawns Rust (argmax then viterbi, **sequentially** — each privstrip process is ~5.6 GB resident, parallel runs OOM and double-up the GPU command queue) and `python-ref/run_reference.py` (argmax + viterbi); compares spans pairwise. Reads Python results from `scripts/.python-cache.jsonl` and auto-populates any missing rows (pass `--no-python` to skip them instead, `--refresh-python` to clear the cache and re-run all). Pass `--js` to also load the transformers.js oracle and emit the full 6-pair matrix.
  - `regression-check.ts` — pre-merge gate: builds the binary, runs `validate.ts --no-python`, exits non-zero if A↔C argmax drops below 96.80% or A↔C viterbi drops below 99.00%. ~2 min on CPU. Use this after any change to `model.rs`/`spans.rs`/`viterbi.rs`.
  - `bench-stream.ts` — repeatable per-row latency benchmark. Same protocol as `validate.ts` but skips comparison. Reports median/p90/p99/throughput from the binary's own `elapsed_us` field. Excludes the cold first row by default.
  - `corpus.jsonl` — 500 English rows from `ai4privacy/pii-masking-300k`, fetched on demand by `validate.ts`.
  - `smoke-stream.ts` — minimal end-to-end smoke test of the stream protocol.
- `VALIDATION.md` — agreement matrix, residual analysis (bf16/f32 drift), and which fixes landed.
- `BENCHMARKS.md` — per-row latency for Rust (CPU + Metal) and Python on the corpus, with per-stage timing breakdown.

## Architecture cheat-sheet

The HF checkpoint is a custom `OpenAIPrivacyFilter` architecture, not a stock HF Transformer:

- 8 layers, `hidden_size = 640`, `intermediate_size = 640`
- GQA: 14 query heads, 2 KV heads, `head_dim = 64`
- Bidirectional sliding-window attention with half-width 128 (so query *i* attends to keys *[i-128, i+128]*); the runtime requires `bidirectional_context = true`
- Sparse MoE: 128 experts, top-4 routing per token, gpt-oss-style SwiGLU with asymmetric clamping
- Attention sinks: one virtual key per head, biased by `sinks[h] * ln(2)` (sinks are stored in log2 units)
- YaRN RoPE with `factor = 32`, `original_max_position_embeddings = 4096`, `rope_theta = 150000`
- Weights are bf16 on disk; we run forward in f32 for op coverage and headroom

`config.num_classes()` reads `num_labels` from `config.json` if present and falls back to 33 (the upstream `openai/privacy-filter` config doesn't declare `num_labels` explicitly). The architecture itself is class-count agnostic — only the classifier head's leading dim changes.

## Build & run

```fish
cargo build --release

# string input (default model is models/base — openai/privacy-filter, 8 PII categories)
target/release/privstrip check -t "Call John at 555-1234"

# multilingual checkpoint with 54 categories and 16 languages
target/release/privstrip check -t "Mein Name ist Klaus Müller" -m models/multilingual

# stream JSONL over stdin
echo '{"id":1,"text":"Call John at 555-1234"}' | target/release/privstrip stream

# Metal (Apple GPU) — off by default because some sandboxes can't init Metal
target/release/privstrip --metal check -f input.txt
```

The default `--model-dir` is `models/base`. The default decoder is `viterbi`; `--decoder argmax` matches transformers.js's per-token-argmax pipeline (see "Decoder choice" below). `--operating-point <name>` selects a Viterbi calibration; the shipped default is `default` (all-zero biases).

### Cargo features

`default = ["apple"]`, and `apple` turns on candle's `accelerate` + `metal` backends. Those link against macOS frameworks and do not build on Linux, so Linux and container builds use `cargo build --release --no-default-features`. No `#[cfg]` in `src/` depends on the feature: candle ships a stub Metal backend, so `--metal` simply errors at runtime when the feature is off. `cargo check --release --no-default-features` on a Mac is a cheap way to confirm the Linux dependency graph still resolves.

## HTTP service (separate repo)

There is no serve mode in this crate. The HTTP service lives in the Timely infra
repo at `services/privstrip` and depends on this crate as a library:

```toml
privstrip = { git = "https://github.com/0rvar/privstrip", rev = "..." }
```

That crate (`privstrip-service`) owns everything deployment-shaped: the axum
server, `/health` and `/v1/detect`, the S3/Hugging Face weights bootstrap and its
`weights-manifest.json`, the request limits, the Dockerfile, and the ECS deploy
script. Its README documents the env contract and the API. **If you change the
library API here, that crate pins a git rev and has to be bumped** — a lib change
is not live until `services/privstrip/Cargo.toml` moves to a new rev and CI
rebuilds the image.

What this crate exposes for it (see `src/lib.rs`): `Engine` (`load`, `detect`
taking a per-call `DecoderMode` + operating point, `count_tokens`,
`operating_points`, `decoder_for`), `DecoderMode`, `DetectedSpan`,
`DetectionResult`, `detection_json`/`detection_error_json` (the per-item envelope
`stream` mode and the HTTP endpoint share, so the two protocols cannot drift),
`pick_device`, a re-exported candle `Device`, and the `config`/`labels`/`model`/
`spans`/`timing`/`viterbi` modules.

## Validation

There are no Rust unit tests — the validation harness is higher-fidelity than anything we'd write inline. See `VALIDATION.md` for the full matrix and residual analysis.

The reference is **C = the Python `opf` package** running the same `model.safetensors` weights. **B = transformers.js** is a secondary oracle of unknown fidelity (different ONNX export, slightly different post-processing). **A = this Rust port** is the system under test.

```fish
nix develop                                         # uv + python311 + cargo + bun
cd python-ref && uv sync && cd ..                   # one-time (downloads torch CPU + opf@main)
cargo build --release

# default: read python from cache, auto-populate any missing rows, no JS
bun scripts/validate.ts --max-rows 500 --matrix-out validation-matrix.json

# skip rows missing from the cache instead of running python (no uv/torch)
bun scripts/validate.ts --max-rows 500 --no-python

# clear the cache and re-run every row through python
bun scripts/validate.ts --max-rows 500 --refresh-python

# full 6-pair matrix including transformers.js oracle
bun scripts/validate.ts --max-rows 500 --js
```

Current baselines on the 500-row corpus (A vs C is the load-bearing comparison):
- `A_viterbi_vs_C_viterbi`: **99.00%** (5 mismatched rows)
- `A_argmax_vs_C_argmax`: **96.80%** (16 mismatched rows)
- `A_viterbi_vs_B`: 93.20% (transformers.js doesn't run our Viterbi — comparing argmax-style spans to Viterbi spans surfaces structural differences, not bugs)

All 21 A↔C mismatches are bf16/f32 precision-tie flips, not algorithmic divergence — `opf` runs most of the attention path at bf16 and we run everything at f32 for op coverage. See `VALIDATION.md` Cluster D for traces.

When `--js` is enabled, `validate.ts` reconstructs token char offsets via `tokenizer.decode([id])` accumulation because transformers.js's `pipeline("token-classification")` does not return `start`/`end`. BPE is reversible, so accumulated decode-lengths give us byte offsets.

## Decoder choice

- **viterbi** (default): constraint-aware. Rejects malformed BIES sequences (e.g. `B-X` followed by `I-Y`) by penalizing illegal transitions to `-inf`. Catches more spans correctly when the model is confident, but disagrees with the upstream library's stock pipeline.
- **argmax**: independent per-token argmax, then hand the BIES tag stream to span aggregation. Matches transformers.js exactly for the per-token decision.

Validation runs both modes to catch regressions in either path.

## What didn't work (and why this matters for future changes)

- **candle-onnx (0.10.x) cannot run the upstream ONNX export.** It's missing `TopK`, `ScatterND`, `GatherND`, `Loop`, and `LayerNormalization`. We load `model.safetensors` directly into hand-written candle ops instead. Don't try to switch to ONNX without first checking whether candle has gained those ops.
- **The official Python `opf` package's HF config differs from the checkpoint's `config.json`.** OPF expects a flat schema (`num_experts`, `bidirectional_context`, flattened `rope_*` keys) while the HF file uses `num_local_experts`, nested `rope_parameters`, etc. `python-ref/run_reference.py::build_opf_config` translates between them and writes a side-car checkpoint dir with the patched config + a symlink to the original `model.safetensors`. If `opf` upstream changes its config schema, that translator is the thing to update.
- **Long inputs:** `build_yarn_rope` caps the precomputed RoPE table at 16384 positions. `Transformer::forward` rejects inputs longer than that with a clear error. To support longer inputs, raise the cap there — the model's `max_position_embeddings` is 131072 so the architecture itself supports more. At long T (≥10k tokens), CPU is *faster* than Metal because each per-layer Metal sync drains a much larger GPU backlog (see BENCHMARKS.md).
- **Closing the bf16/f32 precision gap on A↔C** would require casting Q/K/V and the attention output back to bf16 between RoPE and the einsums, mirroring `opf`'s `sdpa`. Candle 0.10.x bf16 op coverage on CPU is not complete, and matching the recipe would tightly couple the forward pass to the reference. We accept the residual; see `VALIDATION.md` Cluster D.
- **Metal is only ~6–13% faster than CPU on the corpus.** Confirmed root cause: per-layer device→host sync in `MLPBlock::forward`. The gating logits must come back to host so per-expert dispatch can call `Tensor::narrow(0, start, len)` with bucket-aware host-side `usize` arguments. candle 0.10 has **no on-device `topk`, no `nonzero`, and no `narrow` with tensor-shaped bounds** (research: `candle-core-0.10.2/src/sort.rs:260` exists but `topk` does not). Each sync is ~15 ms × 8 layers = 118 ms wasted per forward, dominating Metal latency. candle-transformers' own `mixtral.rs SparseMoeBlock` does the same `to_vec2` host sync. **Masked-dense was prototyped end-to-end and reverted**: composing `arg_sort_last_dim` + `gather` + `scatter_add` for on-device top-k and a single batched `[E, T, H] @ [E, H, 2I]` matmul kills all 8 syncs but the GPU work resurfaces in the next mandatory sync (`logits_sync` jumped 16 → 819 ms/call) — at T ≈ 100 the batched matmul on Apple Silicon doesn't amortize the 32× flop multiplier, and end-to-end median went from 187 ms to 745 ms (4× regression). See BENCHMARKS.md "Why the Metal sync isn't fixed in this revision." Token-replicate + bmm rejected without prototyping (~1.3 GB of replicated expert weights per layer). **Revisit when candle gains on-device topk + tensor-shaped narrow** (both are on candle's main branch but not in 0.10) — true sparse on-device dispatch is the only redesign that wouldn't re-introduce the sync, blow up memory, or do 32× more work for short inputs. First Metal call is still ~11–13 s (kernel compilation + RoPE table upload).

## Performance notes

The sparse MoE forward pass is the hot loop. Notable optimizations already in place:

- Sliding-window attention mask is built **once per forward** in `Transformer::forward` and **cached on the `Transformer`** keyed by T. Repeat token counts hit the cache (Tensor::clone is an Arc bump). Initial per-T construction is the only cost; subsequent forwards at the same length pay nothing.
- MoE expert dispatch builds **one** flat `[T*k]` token-index tensor and one weights tensor per layer, then `narrow()`s per expert (metadata-only, no copy). Without this, Metal would fire ~160 small host→device uploads per layer.
- The unembedding transpose is precomputed once at load (`Transformer::unembedding_t`) instead of materializing `unembedding.t()?.contiguous()?` on every forward.
- Per-expert `mlp1_weight.i(e)?.contiguous()?` calls were dropped — indexing along the leading dim of a contiguous `[num_experts, hidden, ...]` tensor yields a contiguous slice already, so the explicit `.contiguous()` was an unnecessary copy on every active expert per layer per forward.

What's deliberately not optimized yet:
- Top-k routing still happens on CPU. candle 0.10 has no on-device `topk` and `Tensor::narrow` requires host-side `usize` for `start`/`len`, so per-expert dispatch with runtime bucket sizes can't run on-device. See "What didn't work" above for the dead-end research; revisit when candle gains on-device topk.
- Viterbi's inner loop iterates the full 33×33 transition matrix despite ~80% of transitions being `-inf`. Profiling shows decode at 0.05% of CPU and 0.08% of Metal wall time, far below the ~2% threshold for being worth optimizing. Skip until profiling shows it matters.

## Profiling and timing instrumentation

Set `PRIVSTRIP_TIMING=1` to enable per-stage wall-clock instrumentation. The
binary aggregates time spent in tokenize / forward (split into attn /
moe_route / moe_route_sync / moe_expert) / logits (with logits_sync
isolated) / decode / span_extract / serialize, and prints a summary to
stderr at end-of-stream. The fast path is gated on a `OnceLock<bool>` so
the binary pays at most an atomic-load per call site when the env var is
unset.

```fish
PRIVSTRIP_TIMING=1 target/release/privstrip stream -m models < scripts/corpus.jsonl > /dev/null
```

`samply` is in the dev shell. On macOS samply needs Developer Mode enabled
(`sudo DevToolsSecurity -enable`) AND, if running under
`claude-sandbox`, the `(target same-sandbox)` predicate dropped from
`mach-priv-task-port` / `process-info*` (otherwise: `Encountered an error
during profiling: Unknown(1100)`).

## Conventions

- The on-disk model dtype is bf16; we always convert to f32 for the forward pass. Don't change `dtype` in `Transformer::load` without checking that every op supports the new dtype on every device.
- All span byte offsets are in the input text's bytes, not chars. The tokenizer's offset map is byte-indexed.
- `LabelInfo::from_config` parses BIES prefixes from the label string (`"B-private_phone"` → `Boundary::B`, span label `"private_phone"`). The model file is the source of truth for the label set.
- `extract_spans` does whitespace-trim → per-label dedup (keep longest) → greedy non-overlapping select across labels. Changing this order changes the output and will move the validation match rate. Two OPF-parity rules to preserve: (1) a background (`O`) token closes any open span at its left edge — see `labels_to_token_spans` in `src/spans.rs`; (2) `select_non_overlapping` sorts by `(start, -length, label)` so when two spans share a start the longer one wins, then alphabetical by label.
