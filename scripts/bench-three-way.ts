#!/usr/bin/env bun
/**
 * Three-way per-row latency comparison for `privstrip stream`:
 *   - Rust on CPU (viterbi)
 *   - Rust on Metal (viterbi)
 *   - Python `opf` reference (viterbi)
 *
 * All three runs receive the exact same input rows in the same order, and
 * each run is awaited to completion before the next one starts. The first
 * row of every run is excluded from the percentiles (cold-start, especially
 * heavy on Metal due to kernel compilation).
 *
 * Reproduce:
 *   bun scripts/bench-three-way.ts --max-rows 100
 */

import { spawn } from "bun";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";

const HERE = dirname(import.meta.path);
const PROJECT_ROOT = resolve(HERE, "..");

interface Row {
  id: number;
  text: string;
}

interface Reply {
  id: unknown;
  elapsed_us?: number;
  tokens?: number;
  error?: string;
}

interface RunResult {
  label: string;
  cmd: string[];
  rows: number;
  rowsMeasured: number;
  elapsedMs: number[];
  tokens: number[];
  coldFirstRowMs: number | null;
  wallTotalMs: number;
  errors: number;
}

function pct(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx]!;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

async function runOne(label: string, cmd: string[], rows: Row[]): Promise<RunResult> {
  process.stderr.write(`\n=== ${label} ===\nspawning: ${cmd.join(" ")}\n`);
  const proc = spawn({ cmd, stdin: "pipe", stdout: "pipe", stderr: "pipe" });

  void (async () => {
    const reader = (proc.stderr as ReadableStream<Uint8Array>).getReader();
    const dec = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      process.stderr.write(`[${label}] ${dec.decode(value, { stream: true })}`);
    }
  })();

  const decoder = new TextDecoder();
  let buf = "";
  const replies: Reply[] = [];
  const queue: Array<(r: Reply) => void> = [];

  const readerPromise = (async () => {
    const reader = (proc.stdout as ReadableStream<Uint8Array>).getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const r = JSON.parse(line) as Reply;
        replies.push(r);
        const cb = queue.shift();
        if (cb) cb(r);
      }
    }
  })();

  const stdin = proc.stdin as unknown as {
    write: (s: string) => void;
    end: () => void;
    flush?: () => void;
  };

  const wallStart = performance.now();
  for (const row of rows) {
    const p = new Promise<Reply>((res) => queue.push(res));
    stdin.write(JSON.stringify({ id: row.id, text: row.text }) + "\n");
    void stdin.flush?.();
    await p;
  }
  const wallTotalMs = performance.now() - wallStart;

  stdin.end();
  await readerPromise;
  await proc.exited;

  const elapsedMs = replies
    .slice(1)
    .map((r) => (r.elapsed_us ?? 0) / 1000)
    .filter((v) => v > 0);
  const tokens = replies
    .slice(1)
    .map((r) => r.tokens ?? 0)
    .filter((v) => v > 0);

  const errors = replies.filter((r) => r.error).length;
  if (errors) process.stderr.write(`[${label}] ${errors} error rows\n`);

  return {
    label,
    cmd,
    rows: replies.length,
    rowsMeasured: elapsedMs.length,
    elapsedMs,
    tokens,
    coldFirstRowMs: replies[0]?.elapsed_us ? replies[0].elapsed_us / 1000 : null,
    wallTotalMs,
    errors,
  };
}

function fmt(n: number, digits = 1): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function pad(s: string, n: number, right = false): string {
  if (s.length >= n) return s;
  const filler = " ".repeat(n - s.length);
  return right ? s + filler : filler + s;
}

function printComparison(results: RunResult[]) {
  const baseline = results[0]!; // rust-cpu, used for relative-speed column
  const baselineMedian = pct(baseline.elapsedMs, 50);

  const cols = [
    { h: "config", w: 12, right: true },
    { h: "rows", w: 6 },
    { h: "median", w: 10 },
    { h: "p90", w: 10 },
    { h: "p99", w: 10 },
    { h: "mean", w: 10 },
    { h: "req/s", w: 8 },
    { h: "vs cpu", w: 8 },
    { h: "cold", w: 10 },
  ];

  const header = cols.map((c) => pad(c.h, c.w, c.right === true)).join("  ");
  console.log("");
  console.log(header);
  console.log(cols.map((c) => "-".repeat(c.w)).join("  "));

  for (const r of results) {
    const median = pct(r.elapsedMs, 50);
    const reqPerS = r.rows / (r.wallTotalMs / 1000);
    const speedup = baselineMedian / median;
    const row = [
      pad(r.label, cols[0]!.w, true),
      pad(String(r.rowsMeasured), cols[1]!.w),
      pad(fmt(median) + " ms", cols[2]!.w),
      pad(fmt(pct(r.elapsedMs, 90)) + " ms", cols[3]!.w),
      pad(fmt(pct(r.elapsedMs, 99)) + " ms", cols[4]!.w),
      pad(fmt(mean(r.elapsedMs)) + " ms", cols[5]!.w),
      pad(fmt(reqPerS, 2), cols[6]!.w),
      pad(speedup >= 1 ? fmt(speedup, 2) + "×" : "1/" + fmt(1 / speedup, 2) + "×", cols[7]!.w),
      pad(r.coldFirstRowMs == null ? "—" : fmt(r.coldFirstRowMs / 1000, 2) + " s", cols[8]!.w),
    ];
    console.log(row.join("  "));
  }
  console.log("");
  console.log(
    `tokens (median across runs): ${pct(baseline.tokens, 50)}; mean: ${fmt(mean(baseline.tokens), 1)}`,
  );
  console.log(
    `"vs cpu" = (rust-cpu median) / (this median); higher is faster than CPU.`,
  );
}

async function main() {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      corpus: { type: "string", default: join(PROJECT_ROOT, "scripts/corpus.jsonl") },
      bin: { type: "string", default: join(PROJECT_ROOT, "target/release/privstrip") },
      "model-dir": { type: "string", default: join(PROJECT_ROOT, "models") },
      "max-rows": { type: "string", default: "100" },
      "max-bytes": { type: "string", default: "8192" },
      "skip-python": { type: "boolean", default: false },
      "json-out": { type: "string" },
    },
    strict: true,
  });

  const maxRows = Number(values["max-rows"]);
  const maxBytes = Number(values["max-bytes"]);
  const text = await readFile(values.corpus as string, "utf8");
  const rows: Row[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line) as Row;
    if (Buffer.byteLength(r.text, "utf8") > maxBytes) continue;
    rows.push(r);
    if (rows.length >= maxRows) break;
  }
  if (rows.length === 0) throw new Error("no corpus rows after filtering");
  process.stderr.write(`bench-three-way: ${rows.length} rows, viterbi decoder, sequential\n`);

  const bin = values.bin as string;
  const modelDir = values["model-dir"] as string;

  const results: RunResult[] = [];

  results.push(
    await runOne("rust-cpu", [bin, "stream", "-m", modelDir], rows),
  );
  results.push(
    await runOne("rust-metal", [bin, "--metal", "stream", "-m", modelDir], rows),
  );
  if (!values["skip-python"]) {
    results.push(
      await runOne(
        "python-mps",
        [
          "uv",
          "run",
          "--project",
          join(PROJECT_ROOT, "python-ref"),
          "--python",
          "3.11",
          "python",
          join(PROJECT_ROOT, "python-ref/run_reference.py"),
          "stream",
          "-m",
          modelDir,
          "--mps",
        ],
        rows,
      ),
    );
  }

  printComparison(results);

  if (values["json-out"]) {
    const summaries = results.map((r) => ({
      label: r.label,
      rows: r.rows,
      rows_measured: r.rowsMeasured,
      cold_first_row_ms: r.coldFirstRowMs,
      elapsed_ms: {
        median: pct(r.elapsedMs, 50),
        mean: mean(r.elapsedMs),
        p90: pct(r.elapsedMs, 90),
        p99: pct(r.elapsedMs, 99),
        max: r.elapsedMs.length ? Math.max(...r.elapsedMs) : null,
      },
      tokens: {
        median: pct(r.tokens, 50),
        mean: mean(r.tokens),
        p99: pct(r.tokens, 99),
      },
      wall_total_ms: r.wallTotalMs,
      throughput_req_per_s: r.rows / (r.wallTotalMs / 1000),
      cmd: r.cmd,
    }));
    await Bun.write(values["json-out"] as string, JSON.stringify(summaries, null, 2) + "\n");
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
