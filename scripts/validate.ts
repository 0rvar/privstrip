#!/usr/bin/env bun
/**
 * Validate the Rust privstrip port against transformers.js as the oracle.
 *
 * Three things happen here:
 *   1. Corpus is fetched from HF datasets-server (ai4privacy/pii-masking-300k)
 *      on first run and cached to disk.
 *   2. A long-running `privstrip stream` subprocess receives each text.
 *   3. An in-process @huggingface/transformers token-classification pipeline
 *      receives the same text. Its output is normalized and compared.
 *
 * Exit nonzero on any disagreement.
 */

import { spawn } from "bun";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { pipeline, env, type TokenClassificationOutput } from "@huggingface/transformers";

const HERE = dirname(import.meta.path);
const PROJECT_ROOT = resolve(HERE, "..");

type Span = { label: string; byte_start: number; byte_end: number; text: string };

type StreamReply = {
  id: string | number;
  spans?: Span[];
  error?: string;
  elapsed_us?: number;
};

type Stdin = {
  write: (s: string) => number;
  end: () => void;
  flush?: () => void | Promise<void>;
};

class StreamClient {
  private proc: ReturnType<typeof spawn>;
  private decoder = new TextDecoder();
  private buf = "";
  private queue: Array<(reply: StreamReply) => void> = [];
  private stdin: Stdin;
  private readerPromise: Promise<void>;

  constructor(cmd: string[], private label: string) {
    this.proc = spawn({ cmd, stdin: "pipe", stdout: "pipe", stderr: "pipe" });
    this.stdin = this.proc.stdin as unknown as Stdin;
    this.readerPromise = this.consumeStdout();
    this.consumeStderr();
  }

  private async consumeStderr() {
    const reader = (this.proc.stderr as ReadableStream<Uint8Array>).getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (line) process.stderr.write(`[${this.label}] ${line}\n`);
      }
    }
  }

  private async consumeStdout(): Promise<void> {
    const reader = (this.proc.stdout as ReadableStream<Uint8Array>).getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      this.buf += this.decoder.decode(value, { stream: true });
      let nl;
      while ((nl = this.buf.indexOf("\n")) !== -1) {
        const line = this.buf.slice(0, nl);
        this.buf = this.buf.slice(nl + 1);
        if (!line) continue;
        const cb = this.queue.shift();
        if (!cb) {
          process.stderr.write(`[${this.label}] unexpected line: ${line}\n`);
          continue;
        }
        try {
          cb(JSON.parse(line) as StreamReply);
        } catch (e) {
          cb({ id: "<parse>", error: `parse: ${(e as Error).message}: ${line}` });
        }
      }
    }
    for (const cb of this.queue) cb({ id: "<eof>", error: `[${this.label}] stream closed` });
    this.queue = [];
  }

  detect(id: string | number, text: string): Promise<StreamReply> {
    const promise = new Promise<StreamReply>((r) => this.queue.push(r));
    this.stdin.write(JSON.stringify({ id, text }) + "\n");
    void this.stdin.flush?.();
    return promise;
  }

  async close(): Promise<void> {
    this.stdin.end();
    await this.readerPromise;
    await this.proc.exited;
  }
}

type CorpusRow = { id: number; text: string };

async function fetchCorpus(targetCount: number, outPath: string): Promise<void> {
  process.stderr.write(`fetching ${targetCount} rows from ai4privacy/pii-masking-300k...\n`);
  const rows: CorpusRow[] = [];
  let offset = 0;
  const pageSize = 100;
  while (rows.length < targetCount) {
    const url =
      `https://datasets-server.huggingface.co/rows?dataset=ai4privacy%2Fpii-masking-300k` +
      `&config=default&split=validation&offset=${offset}&length=${pageSize}`;
    const r = await fetch(url);
    if (!r.ok) {
      throw new Error(`datasets-server ${r.status}: ${await r.text()}`);
    }
    const data = (await r.json()) as { rows?: Array<{ row: Record<string, unknown> }> };
    const page = data.rows ?? [];
    if (page.length === 0) break;
    for (const item of page) {
      if (item.row.language !== "English") continue;
      const text = item.row.source_text;
      if (typeof text !== "string" || !text.trim()) continue;
      rows.push({ id: rows.length, text });
      if (rows.length >= targetCount) break;
    }
    offset += page.length;
    process.stderr.write(`  ${rows.length}/${targetCount}\r`);
  }
  process.stderr.write(`\n`);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  process.stderr.write(`wrote ${rows.length} rows to ${outPath}\n`);
}

async function loadCorpus(path: string, maxRows: number, maxBytes: number): Promise<CorpusRow[]> {
  const text = await Bun.file(path).text();
  const rows: CorpusRow[] = [];
  let skipped = 0;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line) as CorpusRow;
    if (Buffer.byteLength(r.text, "utf8") > maxBytes) {
      skipped++;
      continue;
    }
    rows.push(r);
    if (rows.length >= maxRows) break;
  }
  if (skipped) process.stderr.write(`skipped ${skipped} rows over ${maxBytes} bytes\n`);
  return rows;
}

function charToByte(text: string, charIdx: number): number {
  return Buffer.byteLength(text.slice(0, charIdx), "utf8");
}

function trimWhitespace(text: string, start: number, end: number): [number, number] {
  while (start < end && /\s/.test(text[start]!)) start++;
  while (end > start && /\s/.test(text[end - 1]!)) end--;
  return [start, end];
}

// transformers.js token-classification doesn't emit start/end offsets, only `word`.
// Locate each match by searching forward from the previous match's end. The Rust
// side trims whitespace from spans, so we do too.
function tjsToSpans(text: string, output: TokenClassificationOutput | TokenClassificationOutput[]): Span[] {
  const arr = (Array.isArray(output[0]) ? output[0] : output) as Array<{
    entity_group?: string;
    entity?: string;
    word: string;
    start?: number;
    end?: number;
  }>;
  const spans: Span[] = [];
  let cursor = 0;
  for (const e of arr) {
    const label = e.entity_group ?? e.entity ?? "";
    if (!label || label === "O") continue;
    let charStart: number;
    let charEnd: number;
    if (e.start != null && e.end != null) {
      charStart = e.start;
      charEnd = e.end;
    } else {
      const word = e.word ?? "";
      const idx = text.indexOf(word, cursor);
      if (idx < 0) continue;
      charStart = idx;
      charEnd = idx + word.length;
    }
    cursor = charEnd;
    [charStart, charEnd] = trimWhitespace(text, charStart, charEnd);
    if (charStart >= charEnd) continue;
    spans.push({
      label,
      byte_start: charToByte(text, charStart),
      byte_end: charToByte(text, charEnd),
      text: text.slice(charStart, charEnd),
    });
  }
  return spans;
}

function spanKey(s: Span): string {
  return `${s.label}@${s.byte_start}..${s.byte_end}`;
}

function compareSpans(a: Span[], b: Span[]) {
  const aKeys = new Set(a.map(spanKey));
  const bKeys = new Set(b.map(spanKey));
  const onlyA = a.filter((s) => !bKeys.has(spanKey(s)));
  const onlyB = b.filter((s) => !aKeys.has(spanKey(s)));
  return { exact: onlyA.length === 0 && onlyB.length === 0, onlyA, onlyB };
}

function snippet(text: string, n = 200): string {
  const flat = text.replace(/\s+/g, " ");
  return flat.length > n ? flat.slice(0, n) + "…" : flat;
}

function fmtSpans(spans: Span[]): string {
  return spans
    .map((s) => `${s.label}@${s.byte_start}..${s.byte_end} ${JSON.stringify(s.text)}`)
    .join("\n      ");
}

function renderProgress(done: number, total: number, exact: number, diff: number, errors: number, startedAt: number) {
  const elapsed = (Date.now() - startedAt) / 1000;
  const rate = done / Math.max(elapsed, 1e-3);
  const eta = Math.max(0, (total - done) / Math.max(rate, 1e-3));
  const width = 30;
  const filled = Math.min(width, Math.floor((width * done) / Math.max(total, 1)));
  const bar = "█".repeat(filled) + "░".repeat(width - filled);
  const line =
    `\r${bar} ${done}/${total}  ` +
    `exact=${exact} diff=${diff} err=${errors}  ` +
    `${rate.toFixed(1)} req/s  eta ${eta.toFixed(0)}s   `;
  process.stderr.write(line);
}

type Args = {
  corpus: string;
  privstripBin: string;
  modelDir: string;
  hfModel: string;
  fetch: number;
  maxRows: number;
  showDiffs: number;
  metal: boolean;
  maxBytes: number;
};

function parseCli(): Args {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      corpus: { type: "string", default: join(PROJECT_ROOT, "scripts/corpus.jsonl") },
      "privstrip-bin": { type: "string", default: join(PROJECT_ROOT, "target/release/privstrip") },
      "model-dir": { type: "string", default: join(PROJECT_ROOT, "models") },
      "hf-model": { type: "string", default: "openai/privacy-filter" },
      fetch: { type: "string", default: "500" },
      "max-rows": { type: "string" },
      "show-diffs": { type: "string", default: "10" },
      "max-bytes": { type: "string", default: "8192" },
      metal: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
    strict: true,
  });
  if (values.help) {
    console.log(`Usage: bun scripts/validate.ts [options]

Options:
  --corpus <path>          JSONL corpus to feed both impls (default scripts/corpus.jsonl)
  --privstrip-bin <path>   Rust binary path (default target/release/privstrip)
  --model-dir <path>       Model dir for the Rust binary (default ./models)
  --hf-model <id>          HF Hub model id for transformers.js (default openai/privacy-filter)
  --fetch <n>              Rows to fetch on first run if corpus is missing (default 500)
  --max-rows <n>           Limit corpus rows used per run
  --max-bytes <n>          Skip rows above this UTF-8 size (default 8192)
  --show-diffs <n>         Print up to this many mismatch examples (default 10)
  --metal                  Pass --metal to the Rust binary
  -h, --help
`);
    process.exit(0);
  }
  return {
    corpus: values.corpus as string,
    privstripBin: values["privstrip-bin"] as string,
    modelDir: values["model-dir"] as string,
    hfModel: values["hf-model"] as string,
    fetch: Number(values.fetch),
    maxRows: values["max-rows"] ? Number(values["max-rows"]) : Infinity,
    showDiffs: Number(values["show-diffs"]),
    metal: values.metal as boolean,
    maxBytes: Number(values["max-bytes"]),
  };
}

async function main() {
  const args = parseCli();

  if (!existsSync(args.corpus)) {
    await fetchCorpus(args.fetch, args.corpus);
  }
  const rows = await loadCorpus(args.corpus, args.maxRows, args.maxBytes);
  process.stderr.write(`loaded ${rows.length} rows from ${args.corpus}\n`);

  // transformers.js looks for:
  //   <localModelPath>/<id>/{config,tokenizer,tokenizer_config}.json
  //   <localModelPath>/<id>/onnx/model.onnx
  // Our `models/` dir already matches that layout, so point it at the parent.
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = dirname(args.modelDir);
  const localId = args.modelDir.split("/").filter(Boolean).slice(-1)[0] ?? "models";
  process.stderr.write(`loading transformers.js (local: ${env.localModelPath}/${localId})...\n`);
  const tjsPipe = await pipeline("token-classification", localId, {
    dtype: "fp32",
    local_files_only: true,
  } as Record<string, unknown>);

  const rust = new StreamClient(
    [
      args.privstripBin,
      "stream",
      "-m",
      args.modelDir,
      ...(args.metal ? ["--metal"] : []),
    ],
    "rust",
  );

  const stats = {
    total: 0,
    exact: 0,
    mismatched: 0,
    rustOnly: 0,
    oracleOnly: 0,
    errors: 0,
    rustUs: 0,
    tjsMs: 0,
  };
  const labelDiffs: Record<string, { rustOnly: number; oracleOnly: number }> = {};
  type Diff = { id: string | number; text: string; rustOnly: Span[]; oracleOnly: Span[] };
  const diffs: Diff[] = [];
  const startedAt = Date.now();

  for (const req of rows) {
    const tjsStart = Date.now();
    const [rustReply, tjsRaw] = await Promise.all([
      rust.detect(req.id, req.text),
      tjsPipe(req.text, { aggregation_strategy: "simple" } as Record<string, unknown>),
    ]);
    stats.tjsMs += Date.now() - tjsStart;
    stats.total++;
    if (rustReply.error) {
      stats.errors++;
      process.stderr.write(`\nerror id=${req.id}: rust=${rustReply.error}\n`);
      continue;
    }
    stats.rustUs += rustReply.elapsed_us ?? 0;
    const oracleSpans = tjsToSpans(req.text, tjsRaw as TokenClassificationOutput);
    const rustSpans = rustReply.spans ?? [];
    const cmp = compareSpans(rustSpans, oracleSpans);
    if (cmp.exact) {
      stats.exact++;
    } else {
      stats.mismatched++;
      stats.rustOnly += cmp.onlyA.length;
      stats.oracleOnly += cmp.onlyB.length;
      for (const s of cmp.onlyA) {
        const e = (labelDiffs[s.label] ??= { rustOnly: 0, oracleOnly: 0 });
        e.rustOnly++;
      }
      for (const s of cmp.onlyB) {
        const e = (labelDiffs[s.label] ??= { rustOnly: 0, oracleOnly: 0 });
        e.oracleOnly++;
      }
      if (diffs.length < args.showDiffs) {
        diffs.push({ id: req.id, text: req.text, rustOnly: cmp.onlyA, oracleOnly: cmp.onlyB });
      }
    }
    renderProgress(stats.total, rows.length, stats.exact, stats.mismatched, stats.errors, startedAt);
  }
  process.stderr.write("\n");

  await rust.close();

  console.log("=== validation summary ===");
  console.log(`total:           ${stats.total}`);
  console.log(`exact match:     ${stats.exact} (${((stats.exact / Math.max(stats.total, 1)) * 100).toFixed(2)}%)`);
  console.log(`mismatched:      ${stats.mismatched}`);
  console.log(`errors:          ${stats.errors}`);
  console.log(`rust-only spans: ${stats.rustOnly}`);
  console.log(`oracle-only:     ${stats.oracleOnly}`);
  console.log(`rust inference:  ${(stats.rustUs / 1e6).toFixed(2)}s`);
  console.log(`oracle inference:${(stats.tjsMs / 1e3).toFixed(2)}s`);

  if (Object.keys(labelDiffs).length) {
    console.log("");
    console.log("by label (rust-only / oracle-only):");
    for (const [label, c] of Object.entries(labelDiffs).sort()) {
      console.log(`  ${label.padEnd(24)} ${String(c.rustOnly).padStart(4)} / ${c.oracleOnly}`);
    }
  }

  if (diffs.length) {
    console.log("");
    console.log(`first ${diffs.length} mismatch(es):`);
    for (const d of diffs) {
      console.log(`  --- id=${d.id} ---`);
      console.log(`    text: ${snippet(d.text)}`);
      if (d.rustOnly.length) console.log(`    rust-only:   ${fmtSpans(d.rustOnly)}`);
      if (d.oracleOnly.length) console.log(`    oracle-only: ${fmtSpans(d.oracleOnly)}`);
    }
  }

  process.exit(stats.mismatched + stats.errors > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
