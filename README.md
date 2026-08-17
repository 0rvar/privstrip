# privstrip

A Rust CLI for detecting personally identifiable information (PII) in text. Runs the [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model directly via [candle](https://github.com/huggingface/candle) + safetensors — no Python, no ONNX runtime.

Two checkpoints from the same model family are supported out of the box:

- **`models/base`** — `openai/privacy-filter`. 8 PII categories, English-trained, the original release.
- **`models/multilingual`** — `OpenMed/privacy-filter-multilingual`. 54 PII categories across 16 languages. Same architecture, HF-standard tensor naming, larger label head. Loaded by the same rust forward pass.

## Quick start

```fish
# 1. Dev shell — provides cargo, rustc, bun, uv, python311, samply.
nix develop

# 2. Pull both checkpoints (~5.6 GB total).
bun scripts/prepare.ts

# 3. Build the rust binary.
cargo build --release

# 4. Run it.
target/release/privstrip check -t "Call John Smith at 555-1234"
target/release/privstrip list  -t "Mein Name ist Klaus Müller" -m models/multilingual
```

The default `--model-dir` is `models/base`; pass `-m models/multilingual` for the 54-category multilingual model.

## CLI modes

```fish
target/release/privstrip <mode> [options]
```

| Mode | Output |
| --- | --- |
| `check` | Print PII locations to stderr; exit 1 if any are found, 0 otherwise. Useful in CI / pre-commit. |
| `redact` | Print the input with PII replaced by `<LABEL>` placeholders. |
| `list` | Print a JSON array of detected spans. |
| `debug` | Per-token (token_id, char range, label, argmax confidence). Set `PRIVSTRIP_DEBUG_TOPK=4` to also dump top-k logits per token. |
| `stream` | Read JSONL `{"id":..., "text":...}` from stdin, emit `{"id":..., "spans":[...]}` per line. Long-running process — load the model once, handle many rows. |

Common flags:

- `-t/--text "..."` — inline input
- `-f/--file <path>` — read input from file
- `-m/--model-dir <dir>` — `models/base` (default) or `models/multilingual`
- `--decoder viterbi|argmax` — Viterbi (default) is constraint-aware; argmax matches transformers.js per-token output
- `--metal` — use Apple GPU (off by default; cold-start is ~12–15 s for kernel compilation)
- `--operating-point <name>` — Viterbi calibration profile from `viterbi_calibration.json`

`stream` is the throughput mode for one-off batch work — every other one-shot mode spins up the model just to handle a single input. Bench results in [BENCHMARKS.md](BENCHMARKS.md).

## HTTP service

This crate is a library plus a CLI. The HTTP service is a separate crate in the
Timely infra repo at `services/privstrip`, which depends on this one:

```toml
privstrip = { git = "https://github.com/0rvar/privstrip", rev = "..." }
```

`privstrip-service` owns the axum server (`GET /health`, `POST /v1/detect`), the
S3-with-Hugging-Face-fallback weights bootstrap, the request limits, the container
image, and the ECS deploy. See `services/privstrip/README.md` in that repo for the
API and the environment contract.

To embed detection in your own Rust program, depend on this crate directly:

```rust
use privstrip::{DecoderMode, Engine, DEFAULT_OPERATING_POINT, pick_device};

let engine = Engine::load(Path::new("models/base"), pick_device(false)?)?;
let result = engine.detect("Call John Smith at 555-1234", DecoderMode::Viterbi, DEFAULT_OPERATING_POINT)?;
for span in &result.spans {
    println!("{} {}..{} {:?}", span.label, span.byte_start, span.byte_end, span.text);
}
```

Span offsets are byte offsets into the input text, not char indices.

## Scripts

All TypeScript scripts run under [bun](https://bun.sh) (included in the dev shell). All Python helpers run under uv with Python 3.11.

| Script | Purpose |
| --- | --- |
| `scripts/prepare.ts` | Download model artifacts (config + tokenizer + safetensors + viterbi calibration) for one or both checkpoints. Skips files whose local size matches upstream; pass `--force` to override. **Run this first.** |
| `scripts/smoke-stream.ts` | Minimal end-to-end smoke test of the `stream` JSONL protocol. ~5 seconds. |
| `scripts/bench-stream.ts` | Per-row latency benchmark (single config). Outputs median/p90/p99/throughput from the binary's own timing field. |
| `scripts/bench-three-way.ts` | Side-by-side latency on the same corpus across three configs: Rust CPU, Rust Metal, Python (OPF) MPS. Reports tok/s and req/s; default is 100 rows. Used to compare base vs multilingual perf. |
| `scripts/validate.ts` | Primary validation harness. Compares Rust (A, both decoders) vs the Python `opf` reference (C). Builds a per-row agreement matrix and prints mismatch examples. Pass `--js` to also load the transformers.js oracle (B) for a 6-pair matrix. |
| `scripts/regression-check.ts` | Pre-merge gate. Builds the binary, runs `validate.ts --no-python`, exits non-zero if A↔C argmax drops below 96.80% or viterbi below 99.00% on the base model. ~2 min on CPU. |
| `scripts/scan-cloudwatch.ts` | Apply privstrip to a sample of CloudWatch log groups via the AWS CLI. Outputs a JSON findings report. Inherits your existing AWS auth. |
| `python-ref/run_reference.py` | The Python reference itself — a JSONL-stream wrapper around the official `opf` package. Translates upstream config / HF-standard config to OPF's schema and rewrites HF-naming safetensors into OPF's layout on first load. Used by `validate.ts` and `bench-three-way.ts`. |

### Running the validation harness

```fish
# One-time setup
cd python-ref && uv sync && cd ..

# Default: A↔C on 500 rows, base model. Python results are cached after the
# first run in scripts/.python-cache-base.jsonl (~25 min cold, instant warm).
bun scripts/validate.ts --max-rows 500

# Same but multilingual.
bun scripts/validate.ts --max-rows 500 --model-dir models/multilingual

# Pre-merge gate (uses the cached python results).
bun scripts/regression-check.ts

# Quick smoke test
bun scripts/smoke-stream.ts
```

### Per-stage timing

The binary supports opt-in per-stage wall-clock instrumentation:

```fish
PRIVSTRIP_TIMING=1 target/release/privstrip stream < scripts/corpus.jsonl > /dev/null
```

It aggregates time across tokenize / forward (attn / moe_route / moe_route_sync / moe_expert) / logits (with logits_sync isolated) / decode / span_extract / serialize, then prints a summary to stderr at end-of-stream. See [BENCHMARKS.md](BENCHMARKS.md) for a per-stage breakdown of where wall time goes.

## Documentation

- [BENCHMARKS.md](BENCHMARKS.md) — per-row latency on the 500-row corpus, Rust CPU vs Metal vs Python, plus a per-stage breakdown.
- [VALIDATION.md](VALIDATION.md) — A↔B↔C agreement matrix, residual analysis (bf16/f32 drift), and which fixes landed.
- [CLAUDE.md](CLAUDE.md) — internal architecture notes: model layout, label/Viterbi pipeline, conventions for changes that affect the forward pass, and dead-end research.

## Layout

```
src/                      — Rust source
  lib.rs                    Library surface (consumed by the infra repo service crate)
  engine.rs                 Engine: tokenize → forward → decode → extract spans
  main.rs                   CLI + I/O + run modes
  model.rs                  Transformer (GQA + sparse MoE + YaRN + bidirectional sliding-window attn)
  viterbi.rs                BIES constraint-aware decoder
  labels.rs                 Config-driven label/boundary metadata
  spans.rs                  token-id → byte-span extraction
  config.rs                 ModelConfig deserialization
  timing.rs                 Opt-in per-stage instrumentation (PRIVSTRIP_TIMING=1)

models/                   — Model artifacts (gitignored; populate via prepare.ts)
  base/                     openai/privacy-filter (33 classes, 8 categories)
  multilingual/             OpenMed/privacy-filter-multilingual (217 classes, 54 categories, 16 langs)

python-ref/               — Python validation reference (uv project)
  run_reference.py          JSONL-stream wrapper around opf

scripts/                  — Bun TypeScript harnesses (see table above)
flake.nix                 — Dev shell (cargo, rustc, bun, uv, python311, samply)
```
