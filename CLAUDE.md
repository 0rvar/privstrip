# privstrip

A Rust CLI for detecting personally identifiable information (PII) in text using the [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model. Runs the model directly via candle + safetensors — no Python, no ONNX runtime.

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
  - `main.rs` — CLI, I/O, run modes
  - `model.rs` — custom transformer architecture (GQA + sparse MoE + YaRN RoPE + bidirectional sliding-window attention)
  - `viterbi.rs` — constraint-aware BIES decoder
  - `labels.rs` — config-driven label/boundary metadata
  - `spans.rs` — token-id → byte-span extraction, whitespace trimming, dedup
  - `config.rs` — model-config deserialization
- `models/` — model artifacts (gitignored)
  - `model.safetensors` (~2.8 GB bf16) — the weights we actually use
  - `tokenizer.json`, `tokenizer_config.json` — o200k_base BPE
  - `config.json` — architecture hyperparameters and `id2label`
  - `viterbi_calibration.json` — Viterbi transition biases. Auto-loaded at startup; the shipped `default` operating point has all-zero biases (a no-op vs constraint-only decoding). Override with `--operating-point <name>`.
  - `onnx/` — ONNX export from upstream (kept for the validation oracle to load via transformers.js, see below). Candle cannot run the ONNX (see "What didn't work")
- `python-ref/` — Python reference (oracle C). Standalone uv project that wraps the official `opf` package and exposes the same JSONL stream protocol as `privstrip stream`. See `python-ref/README.md`.
- `flake.nix` — dev shell providing cargo + rustc + uv + python311 + bun.
- `scripts/`
  - `validate.ts` — primary harness. Spawns Rust (argmax + viterbi) and `python-ref/run_reference.py` (argmax + viterbi); compares spans pairwise. Reads Python results from `scripts/.python-cache.jsonl` and auto-populates any missing rows (pass `--no-python` to skip them instead, `--refresh-python` to clear the cache and re-run all). Pass `--js` to also load the transformers.js oracle and emit the full 6-pair matrix.
  - `corpus.jsonl` — 500 English rows from `ai4privacy/pii-masking-300k`, fetched on demand by `validate.ts`.
  - `smoke-stream.ts` — minimal end-to-end smoke test of the stream protocol.
- `VALIDATION.md` — agreement matrix, residual analysis (bf16/f32 drift), and which fixes landed.
- `BENCHMARKS.md` — per-row latency for Rust (CPU + Metal) and Python on the corpus.

## Architecture cheat-sheet

The HF checkpoint is a custom `OpenAIPrivacyFilter` architecture, not a stock HF Transformer:

- 8 layers, `hidden_size = 640`, `intermediate_size = 640`
- GQA: 14 query heads, 2 KV heads, `head_dim = 64`
- Bidirectional sliding-window attention with half-width 128 (so query *i* attends to keys *[i-128, i+128]*); the runtime requires `bidirectional_context = true`
- Sparse MoE: 128 experts, top-4 routing per token, gpt-oss-style SwiGLU with asymmetric clamping
- Attention sinks: one virtual key per head, biased by `sinks[h] * ln(2)` (sinks are stored in log2 units)
- YaRN RoPE with `factor = 32`, `original_max_position_embeddings = 4096`, `rope_theta = 150000`
- Weights are bf16 on disk; we run forward in f32 for op coverage and headroom

`config.num_classes()` is hard-coded to 33 — the architecture's BIES tag set is fixed for this model family.

## Build & run

```fish
cargo build --release

# string input
target/release/privstrip check -t "Call John at 555-1234" -m models

# stream JSONL over stdin
echo '{"id":1,"text":"Call John at 555-1234"}' | target/release/privstrip stream -m models

# Metal (Apple GPU) — off by default because some sandboxes can't init Metal
target/release/privstrip --metal check -f input.txt -m models
```

The default `--model-dir` is `models`. The default decoder is `viterbi`; `--decoder argmax` matches transformers.js's per-token-argmax pipeline (see "Decoder choice" below). `--operating-point <name>` selects a Viterbi calibration; the shipped default is `default` (all-zero biases).

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
- **Long inputs:** `build_yarn_rope` caps the precomputed RoPE table at 8192 positions. `Transformer::forward` rejects inputs longer than that with a clear error. To support longer inputs, raise the cap there — the model's `max_position_embeddings` is 131072 so the architecture itself supports more.
- **Closing the bf16/f32 precision gap on A↔C** would require casting Q/K/V and the attention output back to bf16 between RoPE and the einsums, mirroring `opf`'s `sdpa`. Candle 0.10.x bf16 op coverage on CPU is not complete, and matching the recipe would tightly couple the forward pass to the reference. We accept the residual; see `VALIDATION.md` Cluster D.
- **Metal is only ~6–13% faster than CPU on the corpus.** The MoE expert dispatch needs CPU-side top-k routing, forcing a device→host sync 8 times per forward. With ~100-token median rows, per-kernel GPU launch overhead also eats the gain. First Metal call is ~10–13 s (kernel compilation + RoPE table upload). Larger inputs would amortize this better, but we have not benchmarked them.

## Performance notes

The sparse MoE forward pass is the hot loop. Notable optimizations already in place:

- Sliding-window attention mask is built **once per forward** in `Transformer::forward` and passed to all 8 attention blocks, not rebuilt per block.
- MoE expert dispatch builds **one** flat `[T*k]` token-index tensor and one weights tensor per layer, then `narrow()`s per expert (metadata-only, no copy). Without this, Metal would fire ~160 small host→device uploads per layer.

What's deliberately not optimized yet:
- Top-k routing still happens on CPU. With candle's current API there's no fused MoE op, and per-expert dispatch needs token-to-expert assignments on the host anyway. One device→host sync per layer is unavoidable without a custom kernel.
- Viterbi's inner loop iterates the full 33×33 transition matrix despite ~80% of transitions being `-inf`. Fast enough for typical inputs (<1 ms for <1k tokens); revisit if profiling shows it.

## Conventions

- The on-disk model dtype is bf16; we always convert to f32 for the forward pass. Don't change `dtype` in `Transformer::load` without checking that every op supports the new dtype on every device.
- All span byte offsets are in the input text's bytes, not chars. The tokenizer's offset map is byte-indexed.
- `LabelInfo::from_config` parses BIES prefixes from the label string (`"B-private_phone"` → `Boundary::B`, span label `"private_phone"`). The model file is the source of truth for the label set.
- `extract_spans` does whitespace-trim → per-label dedup (keep longest) → greedy non-overlapping select across labels. Changing this order changes the output and will move the validation match rate. Two OPF-parity rules to preserve: (1) a background (`O`) token closes any open span at its left edge — see `labels_to_token_spans` in `src/spans.rs`; (2) `select_non_overlapping` sorts by `(start, -length, label)` so when two spans share a start the longer one wins, then alphabetical by label.
