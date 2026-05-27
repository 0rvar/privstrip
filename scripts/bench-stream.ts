#!/usr/bin/env bun
/**
 * Repeatable per-row latency benchmark for the privstrip stream protocol.
 *
 * Pipes a slice of the validation corpus through `privstrip stream` and reports
 * median, p90, p99, mean and total throughput. Excludes the first row (cold)
 * unless --include-cold is passed. The output table is the regression check
 * for the optimization pass — same script, same corpus, same limit, run before
 * and after each change.
 */

import { spawn } from "bun";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";

const HERE = dirname(import.meta.path);
const PROJECT_ROOT = resolve(HERE, "..");

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

async function main() {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      corpus: { type: "string", default: join(PROJECT_ROOT, "scripts/corpus.jsonl") },
      bin: { type: "string", default: join(PROJECT_ROOT, "target/release/privstrip") },
      "model-dir": { type: "string", default: join(PROJECT_ROOT, "models/base") },
      decoder: { type: "string", default: "viterbi" },
      "max-rows": { type: "string", default: "500" },
      "max-bytes": { type: "string", default: "8192" },
      metal: { type: "boolean", default: false },
      "include-cold": { type: "boolean", default: false },
      label: { type: "string" },
      "json-out": { type: "string" },
    },
    strict: true,
  });

  const maxRows = Number(values["max-rows"]);
  const maxBytes = Number(values["max-bytes"]);
  const text = await readFile(values.corpus as string, "utf8");
  const rows: Array<{ id: number; text: string }> = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line) as { id: number; text: string };
    if (Buffer.byteLength(r.text, "utf8") > maxBytes) continue;
    rows.push(r);
    if (rows.length >= maxRows) break;
  }
  if (rows.length === 0) throw new Error("no corpus rows");

  const cmd = [
    values.bin as string,
    "stream",
    "-m",
    values["model-dir"] as string,
    "--decoder",
    values.decoder as string,
    ...(values.metal ? ["--metal"] : []),
  ];
  process.stderr.write(`spawning: ${cmd.join(" ")}\n`);

  const proc = spawn({ cmd, stdin: "pipe", stdout: "pipe", stderr: "pipe" });

  // Drain stderr without blocking.
  void (async () => {
    const reader = (proc.stderr as ReadableStream<Uint8Array>).getReader();
    const dec = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      process.stderr.write(`[child] ${dec.decode(value, { stream: true })}`);
    }
  })();

  const decoder = new TextDecoder();
  let buf = "";
  const replies: Array<{ id: number; elapsed_us?: number; tokens?: number; spans?: unknown[]; error?: string }> = [];
  const queue: Array<(reply: typeof replies[number]) => void> = [];

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
        const r = JSON.parse(line);
        replies.push(r);
        const cb = queue.shift();
        if (cb) cb(r);
      }
    }
  })();

  const stdin = proc.stdin as unknown as { write: (s: string) => void; end: () => void; flush?: () => void };

  const wallStart = performance.now();
  for (const row of rows) {
    const p = new Promise<typeof replies[number]>((res) => queue.push(res));
    stdin.write(JSON.stringify({ id: row.id, text: row.text }) + "\n");
    void stdin.flush?.();
    await p;
  }
  const wallTotalMs = performance.now() - wallStart;

  stdin.end();
  await readerPromise;
  await proc.exited;

  const startIdx = (values["include-cold"] ? 0 : 1);
  const elapsedMs = replies.slice(startIdx)
    .map((r) => (r.elapsed_us ?? 0) / 1000)
    .filter((v) => v > 0);
  const tokens = replies.slice(startIdx).map((r) => r.tokens ?? 0).filter((v) => v > 0);

  const errs = replies.filter((r) => r.error);
  if (errs.length) {
    process.stderr.write(`errors: ${errs.length}\n`);
    for (const e of errs.slice(0, 3)) process.stderr.write(`  ${JSON.stringify(e)}\n`);
  }

  const summary = {
    label: values.label ?? null,
    rows: replies.length,
    rows_measured: elapsedMs.length,
    decoder: values.decoder,
    metal: values.metal ?? false,
    cold_first_row_ms: replies[0]?.elapsed_us ? replies[0].elapsed_us / 1000 : null,
    elapsed_ms: {
      median: pct(elapsedMs, 50),
      mean: mean(elapsedMs),
      p90: pct(elapsedMs, 90),
      p99: pct(elapsedMs, 99),
      max: Math.max(...elapsedMs),
    },
    tokens: {
      median: pct(tokens, 50),
      mean: mean(tokens),
      p99: pct(tokens, 99),
    },
    wall_total_ms: wallTotalMs,
    throughput_req_per_s: replies.length / (wallTotalMs / 1000),
  };

  console.log(JSON.stringify(summary, null, 2));

  if (values["json-out"]) {
    await Bun.write(values["json-out"] as string, JSON.stringify(summary, null, 2) + "\n");
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
