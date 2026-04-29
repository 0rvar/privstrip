# Benchmarks

Per-row latency on the 500-row English corpus
(`scripts/corpus.jsonl`, sampled from `ai4privacy/pii-masking-300k`,
each row ≤ 8 KiB UTF-8) on a single Apple Silicon machine.

The Rust binary is built with `cargo build --release` (`opt-level = 3`,
`lto = "thin"`, `debug = "line-tables-only"`). Latencies reported by the
binary's own `elapsed_us` field; first-call cold-start excluded for Metal
numbers and reported separately.

The harness is `scripts/bench-stream.ts`, which feeds the corpus through
`privstrip stream` over stdin/stdout in sequence (one row at a time).
Reproduce with e.g.:

```fish
bun scripts/bench-stream.ts --max-rows 500 --decoder viterbi --metal --json-out bench.json
```

## Throughput (steady-state, 499 rows after cold)

| Implementation | Decoder | Device | Median | p90 | p99 | Throughput | Cold first row |
|---|---|---|---|---|---|---|---|
| Rust `privstrip stream` | argmax  | CPU   | 207 ms | 245 ms | 270 ms | 4.75 req/s | 186 ms |
| Rust `privstrip stream` | viterbi | CPU   | 207 ms | 246 ms | 274 ms | 4.71 req/s | 195 ms |
| Rust `privstrip stream --metal` | argmax  | Apple GPU | 187 ms | 213 ms | 229 ms | 4.71 req/s | 11.7 s |
| Rust `privstrip stream --metal` | viterbi | Apple GPU | 186 ms | 213 ms | 228 ms | 4.71 req/s | 11.6 s |
| Python `opf` (PyTorch CPU) | argmax  | CPU | ~2.1 s | — | — | ~0.47 req/s | — |
| Python `opf` (PyTorch CPU) | viterbi | CPU | ~2.1 s | — | — | ~0.47 req/s | — |

The Rust port is roughly two orders of magnitude faster per row than the
Python reference. On steady-state Metal vs CPU is a ~10% improvement, far
below what a 50M-active-param MoE on Apple Silicon should hit. The cause
is a candle 0.10 limitation discussed below; the optimization pass
documented it rather than working around it.

For the **previous validate.ts harness that ran argmax + viterbi privstrip
processes in parallel**, both processes shared the GPU command queue and
serialized at the kernel scheduler — but the steady-state per-row
numbers were unchanged once measured with one process at a time, so the
contention was on memory pressure (each f32 model is ~5.6 GB resident),
not GPU throughput. validate.ts now spawns rust subprocesses
sequentially.

## What dominates: per-stage breakdown (PRIVSTRIP_TIMING=1)

The binary supports stage-level wall-clock instrumentation through an
opt-in env var. Reproduce with:

```fish
PRIVSTRIP_TIMING=1 target/release/privstrip stream -m models < scripts/corpus.jsonl > /dev/null
```

Steady-state on 200 corpus rows, viterbi decoder:

### CPU

| Stage                     | Total ms | Calls | ms/call | % top-level |
|---------------------------|---------:|------:|--------:|------------:|
| tokenize                  |     28.0 |   200 |   0.14  |   0.1%      |
| forward                   |  40930   |   200 | 204.6   |  99.8%      |
| ¦ forward.attn            |   4190   |  1600 |   2.62  |             |
| ¦ forward.moe_route       |    307   |  1600 |   0.19  |             |
| ¦ ¦ forward.moe_route_sync|     14.1 |  1600 |   0.009 |             |
| ¦ forward.moe_expert      |  36033   |  1600 |  22.5   |             |
| logits                    |      4.2 |   200 |   0.02  |   0.0%      |
| decode                    |     22.0 |   200 |   0.11  |   0.1%      |
| span_extract              |      1.5 |   200 |   0.008 |   0.0%      |
| serialize                 |     11.9 |   200 |   0.06  |   0.0%      |

CPU is bound by per-expert MoE matmuls. On CPU the device→host "sync" is
just a memcpy of the gating logits, so there is no Metal-style sync penalty.

### Metal

| Stage                     | Total ms | Calls | ms/call | % top-level |
|---------------------------|---------:|------:|--------:|------------:|
| tokenize                  |     44.6 |   200 |   0.22  |   0.1%      |
| forward                   |  45366   |   200 | 226.8   |  93.3%      |
| ¦ forward.attn            |  11010   |  1600 |   6.88  |             |
| ¦ forward.moe_route       |  23864   |  1600 |  14.91  |             |
| ¦ ¦ forward.moe_route_sync|  23619   |  1600 |  14.76  |             |
| ¦ forward.moe_expert      |   9948   |  1600 |   6.22  |             |
| logits                    |   3184   |   200 |  15.92  |   6.5%      |
| ¦ logits_sync             |   3176   |   200 |  15.88  |             |
| decode                    |     38.4 |   200 |   0.19  |   0.1%      |
| span_extract              |      2.0 |   200 |   0.01  |   0.0%      |
| serialize                 |      8.9 |   200 |   0.04  |   0.0%      |

**The MoE per-layer device→host sync is 52% of Metal forward time** —
14.76 ms × 8 layers = 118 ms wasted per request just waiting for the
GPU pipeline to drain so we can read the gating logits to host for
top-k routing. The actual GPU compute (attn + per-expert matmuls) is
about 21 ms × 8 = 168 ms, of which the experts alone are 50 ms. If the
8 syncs went away, Metal latency would be ~70 ms per request — about
2.5× the current ~187 ms — which is roughly what one would expect from
a sparse-MoE this size on Apple Silicon.

The single end-of-forward `to_vec1` for log-softmax also costs one sync
(~16 ms) but that one is unavoidable: decoding has to land on host eventually.

### Why the Metal sync isn't fixed in this revision

candle 0.10.2 has no on-device `topk`, no `nonzero`, and `Tensor::narrow`
takes host-side `usize` for `start`/`len`. Per-expert dispatch with
runtime-shaped buckets requires reading the assignments to host, which
is the sync. Even candle-transformers' own `mixtral.rs` `SparseMoeBlock`
does the same `to_vec2` host sync.

**Masked-dense MoE eval was prototyped and measured** (compose
`arg_sort_last_dim` + `narrow` + `gather` for on-device top-k, build a
`[T, E]` routing mask via `scatter_add`, then run all 128 experts with a
single batched matmul on a replicated `[E, T, H]` input). The bet was
"32× more flops is cheaper than 8 syncs." It wasn't, by a wide margin:

| Path | Median (Metal, viterbi, 500 rows) |
|---|---|
| Sparse + 8 syncs (current) | 187 ms |
| Masked-dense, no syncs | 745 ms |

Per-stage timing showed the sync didn't disappear, it migrated: with
the 8 in-forward syncs gone, all the GPU work coalesced into the next
mandatory sync (the end-of-forward `to_vec1` for log-softmax), where
`logits_sync` jumped from 16 ms/call to 819 ms/call. The dense path's
batched matmuls — `[128, T, H] @ [128, H, 2I]` at the corpus's typical
T ≈ 100 — don't pack well enough on Apple Silicon to amortize the 32×
flop multiplier; the contiguous replication of `normed` to `[E, T, H]`
is also non-trivial (~62 MiB f32 per layer of pure copy). At larger T
the dense path may swing the other way, but the corpus is short-input
heavy.

Token-replicate + bmm-with-gathered-weights was rejected before
prototyping: it would materialize ~1.3 GB of expert weights per layer.

Net: the per-layer host sync stays. Revisit when candle gains on-device
`topk` and tensor-shaped `narrow` (both are on candle's main branch but
not in the 0.10 line we pin to) — true sparse on-device dispatch is the
only redesign that wouldn't either re-introduce the sync, blow up
memory, or do 32× more work for shapes where it doesn't pay.

## Long inputs

The precomputed YaRN RoPE table was raised from 8192 to **16384**
positions. The model architecture supports up to 131072.

A synthetic ~14.8k-token input (corpus rows concatenated):

| Implementation | Decoder | Device | Total wall | tokens | Notes |
|---|---|---|---|---|---|
| Rust | viterbi | CPU       | 342 s | 14856 | forward.attn = 98% (O(T²)) |
| Rust | viterbi | Apple GPU | 446 s | 14856 | forward.moe_route_sync = 97% |

At long T, **CPU is faster than Metal**: each per-layer Metal sync
must drain a much larger backlog, so the 8 syncs per forward
balloon to ~54 s each. CPU has no sync barrier and just does the
math. For long-input batch jobs prefer CPU; for short-input
interactive use the difference is small either way.

## Notes

- All numbers in this file were captured on the same
  Apple Silicon machine in one session. Cross-session bench
  variance has been observed at ±5% for CPU steady-state, dominated
  by thermal state and background load. Treat differences smaller
  than 5% as noise unless reproduced across multiple sessions.
- The harness used to spawn argmax + viterbi privstrip processes in
  parallel; this loaded two copies of the f32 model (~11 GB total)
  and put both into the GPU command queue at once. Sequential is
  the right protocol; the published numbers above use it.
- `bench-stream.ts` excludes the cold first row by default. Cold
  starts on Metal include kernel compilation and RoPE table upload
  (~11–13 s, bigger than warm runtime by orders of magnitude).
