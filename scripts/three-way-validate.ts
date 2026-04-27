#!/usr/bin/env bun
/**
 * Three-way validation: Rust port (A) vs transformers.js oracle (B) vs
 * Python opf reference (C). Produces the full pairwise agreement matrix the
 * project README asks for.
 *
 * The Python leg is slow (CPU torch on a 1.5B MoE), so its outputs are cached
 * to disk keyed on (corpus path, decoder). Use --no-python to skip it once
 * cached. Rust and JS are re-run every invocation (both are <10ms/row).
 *
 * Output:
 *
 *   exact-match by pair, mismatch counts, by-label diff buckets, and a few
 *   examples per cluster.
 */

import { spawn } from "bun";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import {
  AutoTokenizer,
  pipeline,
  env,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

const HERE = dirname(import.meta.path);
const PROJECT_ROOT = resolve(HERE, "..");

type Span = { label: string; byte_start: number; byte_end: number; text: string };

type StreamReply = {
  id: string | number;
  spans?: Span[];
  error?: string;
  elapsed_us?: number;
  decoded_mismatch?: boolean;
  tokens?: number;
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

  constructor(cmd: string[], private label: string, env_?: Record<string, string>) {
    this.proc = spawn({ cmd, stdin: "pipe", stdout: "pipe", stderr: "pipe", env: env_ });
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

function splitBies(entity: string): { prefix: string; label: string } {
  const i = entity.indexOf("-");
  if (i < 0) return { prefix: "", label: entity };
  return { prefix: entity.slice(0, i), label: entity.slice(i + 1) };
}

type RawTok = { entity: string; index: number; score: number; word: string };

async function buildTokenOffsets(
  text: string,
  tokenizer: PreTrainedTokenizer,
): Promise<Array<[number, number]>> {
  const enc = (await tokenizer(text)) as { input_ids: { tolist: () => bigint[][] } };
  const idsRow = enc.input_ids.tolist()[0] ?? [];
  const offsets: Array<[number, number]> = [];
  let cursor = 0;
  for (const id of idsRow) {
    const tokStr = tokenizer.decode([Number(id)]) as unknown as string;
    const len = tokStr.length;
    offsets.push([cursor, cursor + len]);
    cursor += len;
  }
  return offsets;
}

async function tjsToSpans(
  text: string,
  tokenizer: PreTrainedTokenizer,
  raw: RawTok[],
): Promise<Span[]> {
  const flat = await buildTokenOffsets(text, tokenizer);

  const spans: Span[] = [];
  let cur: { label: string; startTok: number; endTok: number } | null = null;

  const finalize = () => {
    if (!cur) return;
    const startPair = flat[cur.startTok];
    const endPair = flat[cur.endTok];
    if (!startPair || !endPair) {
      cur = null;
      return;
    }
    let [cs] = startPair;
    let ce = endPair[1];
    [cs, ce] = trimWhitespace(text, cs, ce);
    if (cs < ce) {
      spans.push({
        label: cur.label,
        byte_start: charToByte(text, cs),
        byte_end: charToByte(text, ce),
        text: text.slice(cs, ce),
      });
    }
    cur = null;
  };

  for (const tok of raw) {
    if (tok.entity === "O" || !tok.entity) {
      finalize();
      continue;
    }
    const { prefix, label } = splitBies(tok.entity);
    const continues = cur && cur.label === label && (prefix === "I" || prefix === "E");
    if (continues) {
      cur!.endTok = tok.index;
    } else {
      finalize();
      cur = { label, startTok: tok.index, endTok: tok.index };
    }
  }
  finalize();
  return spans;
}

function spanKey(s: Span): string {
  return `${s.label}@${s.byte_start}..${s.byte_end}`;
}

function diffSpans(a: Span[], b: Span[]) {
  const aKeys = new Set(a.map(spanKey));
  const bKeys = new Set(b.map(spanKey));
  const onlyA = a.filter((s) => !bKeys.has(spanKey(s)));
  const onlyB = b.filter((s) => !aKeys.has(spanKey(s)));
  return { exact: onlyA.length === 0 && onlyB.length === 0, onlyA, onlyB };
}

type PerRowResult = {
  id: number;
  text: string;
  rust_argmax: Span[];
  rust_viterbi: Span[];
  oracle: Span[];
  python_argmax: Span[];
  python_viterbi: Span[];
  python_argmax_decoded_mismatch: boolean;
  python_viterbi_decoded_mismatch: boolean;
};

type PairKey =
  | "A_argmax_vs_C_argmax"
  | "A_viterbi_vs_C_viterbi"
  | "B_vs_C_argmax"
  | "A_argmax_vs_B"
  | "A_viterbi_vs_B"
  | "B_vs_C_viterbi";

const PAIR_DESCRIPTIONS: Record<PairKey, string> = {
  A_argmax_vs_C_argmax: "Rust argmax vs Python argmax",
  A_viterbi_vs_C_viterbi: "Rust viterbi vs Python viterbi",
  B_vs_C_argmax: "transformers.js vs Python argmax",
  B_vs_C_viterbi: "transformers.js vs Python viterbi",
  A_argmax_vs_B: "Rust argmax vs transformers.js",
  A_viterbi_vs_B: "Rust viterbi vs transformers.js",
};

type PairStats = {
  total: number;
  exact: number;
  mismatched: number;
  onlyAcount: number;
  onlyBcount: number;
  byLabel: Record<string, { onlyA: number; onlyB: number }>;
  examples: { id: number; text: string; onlyA: Span[]; onlyB: Span[] }[];
};

function emptyPairStats(): PairStats {
  return {
    total: 0,
    exact: 0,
    mismatched: 0,
    onlyAcount: 0,
    onlyBcount: 0,
    byLabel: {},
    examples: [],
  };
}

function recordPair(
  stats: PairStats,
  row: PerRowResult,
  a: Span[],
  b: Span[],
  showExamples: number,
) {
  stats.total++;
  const cmp = diffSpans(a, b);
  if (cmp.exact) {
    stats.exact++;
    return;
  }
  stats.mismatched++;
  stats.onlyAcount += cmp.onlyA.length;
  stats.onlyBcount += cmp.onlyB.length;
  for (const s of cmp.onlyA) {
    const e = (stats.byLabel[s.label] ??= { onlyA: 0, onlyB: 0 });
    e.onlyA++;
  }
  for (const s of cmp.onlyB) {
    const e = (stats.byLabel[s.label] ??= { onlyA: 0, onlyB: 0 });
    e.onlyB++;
  }
  if (stats.examples.length < showExamples) {
    stats.examples.push({ id: row.id, text: row.text, onlyA: cmp.onlyA, onlyB: cmp.onlyB });
  }
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

type Args = {
  corpus: string;
  privstripBin: string;
  pythonScript: string;
  modelDir: string;
  cache: string;
  maxRows: number;
  maxBytes: number;
  showExamples: number;
  noPython: boolean;
  pythonOnly: boolean;
  refreshCache: boolean;
  matrixOut: string | null;
};

function parseCli(): Args {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      corpus: { type: "string", default: join(PROJECT_ROOT, "scripts/corpus.jsonl") },
      "privstrip-bin": { type: "string", default: join(PROJECT_ROOT, "target/release/privstrip") },
      "python-script": { type: "string", default: join(PROJECT_ROOT, "python-ref/run_reference.py") },
      "model-dir": { type: "string", default: join(PROJECT_ROOT, "models") },
      cache: { type: "string", default: join(PROJECT_ROOT, "scripts/.python-cache.jsonl") },
      "max-rows": { type: "string" },
      "max-bytes": { type: "string", default: "8192" },
      "show-examples": { type: "string", default: "5" },
      "no-python": { type: "boolean", default: false },
      "python-only": { type: "boolean", default: false },
      "refresh-cache": { type: "boolean", default: false },
      "matrix-out": { type: "string" },
      help: { type: "boolean", short: "h", default: false },
    },
    strict: true,
  });
  if (values.help) {
    console.log(`Usage: bun scripts/three-way-validate.ts [options]

Options:
  --corpus <path>           Corpus JSONL (default scripts/corpus.jsonl)
  --privstrip-bin <path>    Rust binary
  --python-script <path>    python-ref/run_reference.py
  --model-dir <path>        models/
  --cache <path>            Python results cache (JSONL: {decoder, id, spans, decoded_mismatch})
  --max-rows <n>            Limit rows
  --max-bytes <n>           Skip rows above this UTF-8 size (default 8192)
  --show-examples <n>       Mismatch examples per pair (default 5)
  --no-python               Skip python; rely on existing cache
  --python-only             Only build the python cache (no comparison)
  --refresh-cache           Re-run python even if cached
  --matrix-out <path>       Write the matrix as JSON to this path
  -h, --help
`);
    process.exit(0);
  }
  return {
    corpus: values.corpus as string,
    privstripBin: values["privstrip-bin"] as string,
    pythonScript: values["python-script"] as string,
    modelDir: values["model-dir"] as string,
    cache: values.cache as string,
    maxRows: values["max-rows"] ? Number(values["max-rows"]) : Infinity,
    maxBytes: Number(values["max-bytes"]),
    showExamples: Number(values["show-examples"]),
    noPython: values["no-python"] as boolean,
    pythonOnly: values["python-only"] as boolean,
    refreshCache: values["refresh-cache"] as boolean,
    matrixOut: (values["matrix-out"] as string | undefined) ?? null,
  };
}

type CachedResult = { decoder: "argmax" | "viterbi"; id: number; spans: Span[]; decoded_mismatch?: boolean };

async function loadCache(path: string): Promise<Map<string, CachedResult>> {
  const out = new Map<string, CachedResult>();
  if (!existsSync(path)) return out;
  const text = await readFile(path, "utf8");
  let bad = 0;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      const v = JSON.parse(line) as CachedResult;
      out.set(`${v.decoder}:${v.id}`, v);
    } catch {
      bad++;
    }
  }
  if (bad) process.stderr.write(`cache: skipped ${bad} unparseable line(s)\n`);
  return out;
}

async function runPythonForDecoder(
  rows: CorpusRow[],
  decoder: "argmax" | "viterbi",
  args: Args,
  cache: Map<string, CachedResult>,
  cachePath: string,
): Promise<void> {
  const missing = rows.filter((r) => !cache.has(`${decoder}:${r.id}`));
  if (missing.length === 0) {
    process.stderr.write(`python ${decoder}: cache hit for all ${rows.length} rows\n`);
    return;
  }
  process.stderr.write(`python ${decoder}: ${missing.length}/${rows.length} need running\n`);

  const py = new StreamClient(
    [
      "uv",
      "run",
      "--project",
      join(PROJECT_ROOT, "python-ref"),
      "--python",
      "3.11",
      "python",
      args.pythonScript,
      "stream",
      "-m",
      args.modelDir,
      "--decoder",
      decoder,
    ],
    `python-${decoder}`,
  );

  await mkdir(dirname(cachePath), { recursive: true });
  const fs = await import("node:fs/promises");
  // Append-only: never truncate. Each line is a self-contained JSON record. The
  // loader skips unparseable lines (handles a partial trailing line from a
  // crash).
  const fh = await fs.open(cachePath, "a");

  const startedAt = Date.now();
  let done = 0;
  for (const row of missing) {
    const reply = await py.detect(row.id, row.text);
    if (reply.error) {
      process.stderr.write(`\npython error id=${row.id}: ${reply.error}\n`);
      continue;
    }
    const entry: CachedResult = {
      decoder,
      id: row.id,
      spans: reply.spans ?? [],
      decoded_mismatch: reply.decoded_mismatch ?? false,
    };
    cache.set(`${decoder}:${row.id}`, entry);
    await fh.write(JSON.stringify(entry) + "\n");
    done++;
    if (done % 5 === 0 || done === missing.length) {
      const elapsed = (Date.now() - startedAt) / 1000;
      const rate = done / Math.max(elapsed, 1e-3);
      const eta = (missing.length - done) / Math.max(rate, 1e-3);
      process.stderr.write(
        `\rpython ${decoder}: ${done}/${missing.length}  ${rate.toFixed(2)} req/s  eta ${eta.toFixed(0)}s   `,
      );
    }
  }
  process.stderr.write("\n");
  await py.close();
  await fh.close();
}

async function main() {
  const args = parseCli();

  const rows = await loadCorpus(args.corpus, args.maxRows, args.maxBytes);
  process.stderr.write(`loaded ${rows.length} rows from ${args.corpus}\n`);

  const cache = await loadCache(args.cache);

  if (!args.noPython) {
    if (args.refreshCache) {
      cache.clear();
      const fs = await import("node:fs/promises");
      try { await fs.unlink(args.cache); } catch {}
    }
    await runPythonForDecoder(rows, "argmax", args, cache, args.cache);
    await runPythonForDecoder(rows, "viterbi", args, cache, args.cache);
  } else {
    process.stderr.write(`--no-python: relying on cache (${cache.size} entries)\n`);
  }

  if (args.pythonOnly) {
    process.stderr.write(`--python-only: cache populated, skipping comparison\n`);
    return;
  }

  // Set up transformers.js oracle.
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = dirname(args.modelDir);
  const localId = args.modelDir.split("/").filter(Boolean).slice(-1)[0] ?? "models";
  process.stderr.write(`loading transformers.js (local: ${env.localModelPath}/${localId})...\n`);
  const tjsPipe = await pipeline("token-classification", localId, {
    dtype: "fp32",
    local_files_only: true,
  } as Record<string, unknown>);
  const tjsTokenizer = (await AutoTokenizer.from_pretrained(localId, {
    local_files_only: true,
  } as never)) as PreTrainedTokenizer;

  const rustArgmax = new StreamClient(
    [args.privstripBin, "stream", "-m", args.modelDir, "--decoder", "argmax"],
    "rust-argmax",
  );
  const rustViterbi = new StreamClient(
    [args.privstripBin, "stream", "-m", args.modelDir, "--decoder", "viterbi"],
    "rust-viterbi",
  );

  const stats: Record<PairKey, PairStats> = {
    A_argmax_vs_C_argmax: emptyPairStats(),
    A_viterbi_vs_C_viterbi: emptyPairStats(),
    B_vs_C_argmax: emptyPairStats(),
    B_vs_C_viterbi: emptyPairStats(),
    A_argmax_vs_B: emptyPairStats(),
    A_viterbi_vs_B: emptyPairStats(),
  };

  const startedAt = Date.now();
  let done = 0;
  let pythonMismatches = 0;
  for (const row of rows) {
    const pyA = cache.get(`argmax:${row.id}`);
    const pyV = cache.get(`viterbi:${row.id}`);
    if (!pyA || !pyV) {
      process.stderr.write(`\npython cache miss for id=${row.id}; skipping row\n`);
      continue;
    }
    if (pyA.decoded_mismatch || pyV.decoded_mismatch) {
      pythonMismatches++;
      // We still compare — but spans may refer to the decoded form, not the input.
      // The row will likely show as a mismatch in pairs involving Python; that's expected.
    }

    const [rA, rV, tjsRaw] = await Promise.all([
      rustArgmax.detect(row.id, row.text),
      rustViterbi.detect(row.id, row.text),
      tjsPipe(row.text, { aggregation_strategy: "none" } as Record<string, unknown>),
    ]);

    if (rA.error || rV.error) {
      process.stderr.write(`\nrust error id=${row.id}: argmax=${rA.error} viterbi=${rV.error}\n`);
      continue;
    }
    const oracle = await tjsToSpans(row.text, tjsTokenizer, tjsRaw as RawTok[]);

    recordPair(stats.A_argmax_vs_C_argmax, { ...row, rust_argmax: rA.spans!, rust_viterbi: rV.spans!, oracle, python_argmax: pyA.spans, python_viterbi: pyV.spans, python_argmax_decoded_mismatch: pyA.decoded_mismatch ?? false, python_viterbi_decoded_mismatch: pyV.decoded_mismatch ?? false }, rA.spans!, pyA.spans, args.showExamples);
    recordPair(stats.A_viterbi_vs_C_viterbi, { ...row, rust_argmax: rA.spans!, rust_viterbi: rV.spans!, oracle, python_argmax: pyA.spans, python_viterbi: pyV.spans, python_argmax_decoded_mismatch: pyA.decoded_mismatch ?? false, python_viterbi_decoded_mismatch: pyV.decoded_mismatch ?? false }, rV.spans!, pyV.spans, args.showExamples);
    recordPair(stats.B_vs_C_argmax, { ...row, rust_argmax: rA.spans!, rust_viterbi: rV.spans!, oracle, python_argmax: pyA.spans, python_viterbi: pyV.spans, python_argmax_decoded_mismatch: pyA.decoded_mismatch ?? false, python_viterbi_decoded_mismatch: pyV.decoded_mismatch ?? false }, oracle, pyA.spans, args.showExamples);
    recordPair(stats.B_vs_C_viterbi, { ...row, rust_argmax: rA.spans!, rust_viterbi: rV.spans!, oracle, python_argmax: pyA.spans, python_viterbi: pyV.spans, python_argmax_decoded_mismatch: pyA.decoded_mismatch ?? false, python_viterbi_decoded_mismatch: pyV.decoded_mismatch ?? false }, oracle, pyV.spans, args.showExamples);
    recordPair(stats.A_argmax_vs_B, { ...row, rust_argmax: rA.spans!, rust_viterbi: rV.spans!, oracle, python_argmax: pyA.spans, python_viterbi: pyV.spans, python_argmax_decoded_mismatch: pyA.decoded_mismatch ?? false, python_viterbi_decoded_mismatch: pyV.decoded_mismatch ?? false }, rA.spans!, oracle, args.showExamples);
    recordPair(stats.A_viterbi_vs_B, { ...row, rust_argmax: rA.spans!, rust_viterbi: rV.spans!, oracle, python_argmax: pyA.spans, python_viterbi: pyV.spans, python_argmax_decoded_mismatch: pyA.decoded_mismatch ?? false, python_viterbi_decoded_mismatch: pyV.decoded_mismatch ?? false }, rV.spans!, oracle, args.showExamples);

    done++;
    if (done % 10 === 0 || done === rows.length) {
      const elapsed = (Date.now() - startedAt) / 1000;
      const rate = done / Math.max(elapsed, 1e-3);
      const eta = (rows.length - done) / Math.max(rate, 1e-3);
      process.stderr.write(
        `\rcompare: ${done}/${rows.length}  ${rate.toFixed(1)} req/s  eta ${eta.toFixed(0)}s   `,
      );
    }
  }
  process.stderr.write("\n");

  await rustArgmax.close();
  await rustViterbi.close();

  console.log("=== three-way agreement matrix ===");
  console.log(`corpus rows compared: ${stats.A_argmax_vs_C_argmax.total}`);
  console.log(`python decoded_mismatch rows: ${pythonMismatches}`);
  console.log("");
  const order: PairKey[] = [
    "A_argmax_vs_C_argmax",
    "A_viterbi_vs_C_viterbi",
    "B_vs_C_argmax",
    "B_vs_C_viterbi",
    "A_argmax_vs_B",
    "A_viterbi_vs_B",
  ];
  console.log(
    `${"pair".padEnd(28)} ${"description".padEnd(36)} ${"exact%".padStart(8)} ${"mismatch".padStart(8)} ${"left-only".padStart(9)} ${"right-only".padStart(10)}`,
  );
  for (const k of order) {
    const s = stats[k];
    const pct = ((s.exact / Math.max(s.total, 1)) * 100).toFixed(2);
    console.log(
      `${k.padEnd(28)} ${PAIR_DESCRIPTIONS[k].padEnd(36)} ${pct.padStart(7)}% ${String(s.mismatched).padStart(8)} ${String(s.onlyAcount).padStart(9)} ${String(s.onlyBcount).padStart(10)}`,
    );
  }

  // Per-label diffs and examples for each pair.
  for (const k of order) {
    const s = stats[k];
    if (s.mismatched === 0) continue;
    console.log("");
    console.log(`--- ${k}: by label (left-only / right-only) ---`);
    for (const [label, c] of Object.entries(s.byLabel).sort()) {
      console.log(`  ${label.padEnd(24)} ${String(c.onlyA).padStart(4)} / ${c.onlyB}`);
    }
    if (s.examples.length) {
      console.log(`  first ${s.examples.length} mismatch(es):`);
      for (const d of s.examples) {
        console.log(`    --- id=${d.id} ---`);
        console.log(`      text: ${snippet(d.text)}`);
        if (d.onlyA.length) console.log(`      left-only:  ${fmtSpans(d.onlyA)}`);
        if (d.onlyB.length) console.log(`      right-only: ${fmtSpans(d.onlyB)}`);
      }
    }
  }

  if (args.matrixOut) {
    const matrix: Record<string, unknown> = {};
    for (const k of order) {
      const s = stats[k];
      matrix[k] = {
        description: PAIR_DESCRIPTIONS[k],
        total: s.total,
        exact: s.exact,
        exact_pct: (s.exact / Math.max(s.total, 1)) * 100,
        mismatched: s.mismatched,
        onlyA: s.onlyAcount,
        onlyB: s.onlyBcount,
        by_label: s.byLabel,
      };
    }
    await writeFile(args.matrixOut, JSON.stringify(matrix, null, 2));
    process.stderr.write(`wrote matrix to ${args.matrixOut}\n`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
