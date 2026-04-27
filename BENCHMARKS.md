# Benchmarks

Numbers below are from the 500-row English corpus
(`scripts/corpus.jsonl`, sampled from `ai4privacy/pii-masking-300k`,
each row ≤ 8 KiB UTF-8) on a single Apple Silicon machine,
running CPU-only.

Both numbers exclude model load time. The Rust binary is built with
`cargo build --release` (`opt-level = 3`, `lto = "thin"`).

## Throughput

500 corpus rows on an Apple Silicon machine (aarch64-darwin), median row
~100 tokens, max 8 KiB UTF-8. Latencies reported by the binary's own
`elapsed_us` field; first-call cold-start excluded for Metal numbers.

| Implementation | Decoder | Device | Median latency | Throughput |
|---|---|---|---|---|
| Rust (`privstrip stream`) | argmax  | CPU       | 196 ms | 5.0 req/s |
| Rust (`privstrip stream`) | viterbi | CPU       | 208 ms | 4.8 req/s |
| Rust (`privstrip stream --metal`) | argmax  | Apple GPU | 185 ms (warm) | 5.4 req/s |
| Rust (`privstrip stream --metal`) | viterbi | Apple GPU | 185 ms (warm) | 5.4 req/s |
| Python `opf` (PyTorch CPU) | argmax  | CPU | ~2.1 s | ~0.47 req/s |
| Python `opf` (PyTorch CPU) | viterbi | CPU | ~2.1 s | ~0.47 req/s |

The Rust port is roughly two orders of magnitude faster per row than the
Python reference. The reference exists for correctness, not speed; do not
read these numbers as "torch is slow."

## Notes

- The MoE forward pass is the hot loop in Rust. Per-layer dispatch is
  already batched into a single host→device upload (see CLAUDE.md). On
  CPU, top-k expert routing dominates.
- **Metal is only ~6–13% faster than CPU on this corpus.** The
  expected 5–10× Metal speedup doesn't materialize because the MoE
  expert dispatch needs CPU-side top-k routing per layer, forcing a
  device→host sync 8 times per forward. With ~100-token median rows,
  the per-kernel GPU launch overhead also eats into the gain. Larger
  inputs would amortize that overhead better; we have not benchmarked
  long-input throughput.
- The first Metal call costs ~10–13 s (kernel compilation and RoPE
  table upload). Throughput numbers above are warm — first-call
  latency is excluded.
- Long inputs (>8192 tokens) are rejected by the Rust forward pass —
  the precomputed YaRN RoPE table caps at 8192. The model architecture
  itself supports up to 131072.

## Methodology

```fish
bun scripts/three-way-validate.ts --max-rows 500
```

Per-row latency is the difference between the JSONL request being
written to the subprocess's stdin and the matching reply line being
received. The wall-clock numbers are pessimistic for Rust because
the harness serializes requests one at a time; Rust on its own can
sustain higher throughput with batched stdin feeding.
