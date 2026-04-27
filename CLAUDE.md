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
  - `viterbi_calibration.json` — unused at runtime; kept for reference
  - `onnx/` — ONNX export from upstream (kept for the validation oracle to load via transformers.js, see below). Candle cannot run the ONNX (see "What didn't work")
- `scripts/`
  - `validate.ts` — bun script: spawns the Rust binary in `stream` mode and runs transformers.js in-process on a corpus of HF rows; reports exact-match rate + per-label disagreement
  - `corpus.jsonl` — 500 English rows from `ai4privacy/pii-masking-300k`, fetched on demand by `validate.ts`
  - `smoke-stream.ts` — minimal end-to-end smoke test of the stream protocol

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

The default `--model-dir` is `models`. The default decoder is `viterbi`; `--decoder argmax` matches transformers.js's per-token-argmax pipeline (see "Decoder choice" below).

## Validation

`scripts/validate.ts` is the test harness. There are no Rust unit tests — the validation script is higher-fidelity than anything we'd write inline.

```fish
# fetch corpus on first run, then validate against transformers.js
bun scripts/validate.ts \
  --decoder argmax \
  --privstrip-bin target/release/privstrip \
  --model-dir models
```

Current baselines on the 500-row corpus:
- `--decoder argmax`: 95.60% exact-match against transformers.js
- `--decoder viterbi`: 93.20% exact-match

The remaining gap is **not** forward-pass drift — it's span aggregation rules differing between our `extract_spans` and transformers.js's `aggregation_strategy: "simple"` post-processing. Forward-pass parity with the official ONNX export is essentially confirmed by the argmax run.

The validation script reconstructs token char offsets via `tokenizer.decode([id])` accumulation because transformers.js's `pipeline("token-classification")` does not return `start`/`end`, and its tokenizer also does not return `offset_mapping`. BPE is reversible, so accumulated decode-lengths give us byte offsets.

## Decoder choice

- **viterbi** (default): constraint-aware. Rejects malformed BIES sequences (e.g. `B-X` followed by `I-Y`) by penalizing illegal transitions to `-inf`. Catches more spans correctly when the model is confident, but disagrees with the upstream library's stock pipeline.
- **argmax**: independent per-token argmax, then hand the BIES tag stream to span aggregation. Matches transformers.js exactly for the per-token decision.

Validation runs both modes to catch regressions in either path.

## What didn't work (and why this matters for future changes)

- **candle-onnx (0.10.x) cannot run the upstream ONNX export.** It's missing `TopK`, `ScatterND`, `GatherND`, `Loop`, and `LayerNormalization`. We load `model.safetensors` directly into hand-written candle ops instead. Don't try to switch to ONNX without first checking whether candle has gained those ops.
- **The official Python `opf` package drifted incompatible with this checkpoint.** The HF model file we use was published 4–5 days apart from a github release that no longer loads it (different tokenizer expectations, etc.). The transformers.js path is the only known-working oracle for now.
- **Long inputs:** `build_yarn_rope` caps the precomputed RoPE table at 8192 positions. `Transformer::forward` rejects inputs longer than that with a clear error. To support longer inputs, raise the cap there — the model's `max_position_embeddings` is 131072 so the architecture itself supports more.

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
- `extract_spans` does whitespace-trim → per-label dedup (keep longest) → greedy non-overlapping select across labels. Changing this order changes the output and will move the validation match rate.
