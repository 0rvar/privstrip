//! Lightweight per-stage wall-clock instrumentation behind `PRIVSTRIP_TIMING=1`.
//!
//! Off by default: `time_stage` early-returns the no-op path so the release
//! binary pays at most one atomic-load per call site when the env var is unset.
//! When enabled, each stage's elapsed time and call count accumulate into
//! atomics; `report` prints a summary at end-of-stream.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Stage {
    Tokenize = 0,
    Forward = 1,            // whole Transformer::forward
    ForwardAttn = 2,        // sum over all attention blocks
    ForwardMoeRoute = 3,    // gate_logits + softmax + sort + dispatch metadata upload
    ForwardMoeRouteSync = 4,// just the to_vec2 device->host sync (subset of MoeRoute)
    ForwardMoeExpert = 5,   // per-expert matmul / activation work
    Logits = 6,             // final_norm + unembed + log_softmax + to_vec1 sync
    LogitsSync = 7,         // just the to_vec1 device->host sync (subset of Logits)
    Decode = 8,
    SpanExtract = 9,
    Serialize = 10,
}

const NUM_STAGES: usize = 11;

static STAGE_LABELS: [&str; NUM_STAGES] = [
    "tokenize",
    "forward",
    "  forward.attn",
    "  forward.moe_route",
    "    forward.moe_route_sync",
    "  forward.moe_expert",
    "logits",
    "  logits_sync",
    "decode",
    "span_extract",
    "serialize",
];

// Top-level stages that should sum to roughly the per-row latency. The
// indented stages above are sub-buckets of forward / logits.
const TOPLEVEL: &[Stage] = &[
    Stage::Tokenize,
    Stage::Forward,
    Stage::Logits,
    Stage::Decode,
    Stage::SpanExtract,
    Stage::Serialize,
];

const SUB_OF_FORWARD: &[Stage] = &[
    Stage::ForwardAttn,
    Stage::ForwardMoeRoute,
    Stage::ForwardMoeExpert,
];

static ENABLED: OnceLock<bool> = OnceLock::new();

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static TOTALS_NS: [AtomicU64; NUM_STAGES] = [ZERO; NUM_STAGES];
static COUNTS: [AtomicU64; NUM_STAGES] = [ZERO; NUM_STAGES];

#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PRIVSTRIP_TIMING").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

#[inline]
pub fn time_stage<F, R>(stage: Stage, f: F) -> R
where
    F: FnOnce() -> R,
{
    if !enabled() {
        return f();
    }
    time_stage_slow(stage, f)
}

#[inline(never)]
fn time_stage_slow<F, R>(stage: Stage, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_nanos() as u64;
    TOTALS_NS[stage as usize].fetch_add(elapsed, Ordering::Relaxed);
    COUNTS[stage as usize].fetch_add(1, Ordering::Relaxed);
    result
}

pub fn report<W: std::io::Write>(out: &mut W) -> std::io::Result<()> {
    if !enabled() {
        return Ok(());
    }
    let toplevel_ms: f64 = TOPLEVEL
        .iter()
        .map(|s| TOTALS_NS[*s as usize].load(Ordering::Relaxed) as f64 / 1_000_000.0)
        .sum();
    let forward_ms = TOTALS_NS[Stage::Forward as usize].load(Ordering::Relaxed) as f64 / 1_000_000.0;
    let forward_breakdown_ms: f64 = SUB_OF_FORWARD
        .iter()
        .map(|s| TOTALS_NS[*s as usize].load(Ordering::Relaxed) as f64 / 1_000_000.0)
        .sum();

    writeln!(out, "")?;
    writeln!(out, "=== PRIVSTRIP_TIMING report ===")?;
    writeln!(
        out,
        "{:<28} {:>12} {:>9} {:>10} {:>8}",
        "stage", "total_ms", "count", "ms/call", "% top"
    )?;
    for i in 0..NUM_STAGES {
        let ns = TOTALS_NS[i].load(Ordering::Relaxed);
        let count = COUNTS[i].load(Ordering::Relaxed);
        if count == 0 {
            continue;
        }
        let ms = ns as f64 / 1_000_000.0;
        let avg = ms / count as f64;
        let is_toplevel = TOPLEVEL.iter().any(|s| *s as usize == i);
        let pct_str = if is_toplevel && toplevel_ms > 0.0 {
            format!("{:>7.1}%", ms / toplevel_ms * 100.0)
        } else {
            "".into()
        };
        writeln!(
            out,
            "{:<28} {:>12.2} {:>9} {:>10.3} {:>8}",
            STAGE_LABELS[i], ms, count, avg, pct_str
        )?;
    }
    writeln!(
        out,
        "{:<28} {:>12.2}",
        "toplevel sum (~ wall)", toplevel_ms
    )?;
    if forward_ms > 0.0 && forward_breakdown_ms > 0.0 {
        let coverage = forward_breakdown_ms / forward_ms * 100.0;
        writeln!(
            out,
            "forward sub-bucket coverage: {:>5.1}% ({:.2}ms / {:.2}ms)",
            coverage, forward_breakdown_ms, forward_ms
        )?;
    }
    Ok(())
}
