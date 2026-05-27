#!/usr/bin/env bun
/**
 * Download model artifacts for privstrip from Hugging Face.
 *
 * Pulls both supported checkpoints into models/{base,multilingual}/. Each file
 * is skipped if it already exists with the same byte length as upstream; pass
 * --force to re-download regardless.
 *
 *   bun scripts/prepare.ts                       # both checkpoints
 *   bun scripts/prepare.ts --model base          # just the upstream model
 *   bun scripts/prepare.ts --model multilingual  # just the OpenMed fine-tune
 *   bun scripts/prepare.ts --force               # ignore the size-match skip
 *
 * The OPF reference (python-ref) lazily materializes its own workdir on first
 * use, so we don't pre-stage anything for it here.
 */

import { mkdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";

const HERE = dirname(import.meta.path);
const PROJECT_ROOT = resolve(HERE, "..");

interface Checkpoint {
  name: string;
  repo: string;
  dest: string;
  files: string[];
}

const CHECKPOINTS: Checkpoint[] = [
  // viterbi_calibration.json is handled separately from `files` because
  // upstream publishes it for some checkpoints (base) but not others
  // (multilingual). The post-loop fetch tries the upstream URL, falls back
  // to writing a zero-biases default, and won't clobber an existing local
  // file unless --force is set.
  {
    name: "base",
    repo: "openai/privacy-filter",
    dest: join(PROJECT_ROOT, "models/base"),
    files: [
      "config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "model.safetensors",
    ],
  },
  {
    name: "multilingual",
    repo: "OpenMed/privacy-filter-multilingual",
    dest: join(PROJECT_ROOT, "models/multilingual"),
    files: [
      "config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "model.safetensors",
    ],
  },
];

// All-zero Viterbi transition biases. Equivalent to constraint-only decoding,
// so this is a safe default for any checkpoint in the openai-privacy-filter
// family until a checkpoint-specific calibration has been derived.
const DEFAULT_CALIBRATION = {
  operating_points: {
    default: {
      biases: {
        transition_bias_background_stay: 0.0,
        transition_bias_background_to_start: 0.0,
        transition_bias_end_to_background: 0.0,
        transition_bias_end_to_start: 0.0,
        transition_bias_inside_to_continue: 0.0,
        transition_bias_inside_to_end: 0.0,
      },
    },
  },
};

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MiB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GiB`;
}

function rel(path: string): string {
  return path.startsWith(PROJECT_ROOT + "/") ? path.slice(PROJECT_ROOT.length + 1) : path;
}

async function fileSize(path: string): Promise<number | null> {
  try {
    return (await stat(path)).size;
  } catch {
    return null;
  }
}

async function remoteSize(url: string): Promise<number | null> {
  // HEAD against Hugging Face follows a redirect to the CDN; the final
  // Content-Length is what we need. Some asset endpoints reject HEAD outright
  // and 405; treat that as "size unknown, just re-download."
  const res = await fetch(url, { method: "HEAD", redirect: "follow" });
  if (!res.ok) return null;
  const cl = res.headers.get("content-length");
  return cl ? Number(cl) : null;
}

async function streamDownload(url: string, dest: string): Promise<number> {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) {
    throw new Error(`fetch ${url}: ${res.status} ${res.statusText}`);
  }
  await mkdir(dirname(dest), { recursive: true });
  const total = Number(res.headers.get("content-length") ?? 0);
  // Write to a temp file and rename on success. Bun.file(dest).writer() does
  // not truncate the existing file before writing, so a shorter download into
  // an already-present path leaves stale tail bytes (corrupted JSON, broken
  // safetensors). The tempfile+rename pattern avoids that and also makes a
  // mid-download interrupt non-destructive.
  const tmp = `${dest}.partial`;
  await rm(tmp, { force: true });
  const writer = Bun.file(tmp).writer();
  let written = 0;
  let lastReport = 0;
  const reader = res.body.getReader();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      writer.write(value);
      written += value.length;
      // Progress ticks for files large enough to notice; emit at most every 32 MiB.
      if (total > 64 * 1024 * 1024 && written - lastReport > 32 * 1024 * 1024) {
        const pct = total > 0 ? ((written / total) * 100).toFixed(0) : "?";
        process.stderr.write(
          `        ${rel(dest)}: ${fmtBytes(written)} / ${fmtBytes(total)} (${pct}%)\r`,
        );
        lastReport = written;
      }
    }
    await writer.end();
    if (lastReport > 0) process.stderr.write("\n");
    await rename(tmp, dest);
  } catch (e) {
    await rm(tmp, { force: true });
    throw e;
  }
  return written;
}

async function downloadOne(
  url: string,
  dest: string,
  force: boolean,
  options: { optional?: boolean } = {},
): Promise<"downloaded" | "skipped" | "missing"> {
  const local = await fileSize(dest);
  if (!force && local !== null) {
    const remote = await remoteSize(url);
    if (remote !== null && remote === local) {
      console.log(`  skip  ${rel(dest)}  (${fmtBytes(local)} already present)`);
      return "skipped";
    }
    if (remote === null && options.optional) {
      // HEAD failed and the file is optional; keep what we have.
      console.log(`  skip  ${rel(dest)}  (${fmtBytes(local)} present, upstream unreachable)`);
      return "skipped";
    }
  }

  // Probe to distinguish "upstream doesn't have this file" (404) from a real fetch failure.
  if (options.optional) {
    const probe = await fetch(url, { method: "HEAD", redirect: "follow" });
    if (probe.status === 404) {
      console.log(`  miss  ${rel(dest)}  (not published upstream)`);
      return "missing";
    }
  }

  console.log(`  pull  ${url}`);
  const written = await streamDownload(url, dest);
  console.log(`        wrote ${rel(dest)}  (${fmtBytes(written)})`);
  return "downloaded";
}

async function prepareCheckpoint(cp: Checkpoint, force: boolean): Promise<void> {
  console.log(`\n[${cp.name}]  ${cp.repo}  →  ${rel(cp.dest)}/`);
  await mkdir(cp.dest, { recursive: true });

  for (const f of cp.files) {
    const url = `https://huggingface.co/${cp.repo}/resolve/main/${f}`;
    await downloadOne(url, join(cp.dest, f), force);
  }

  // Calibration: prefer the upstream file if it exists; otherwise drop a
  // zero-biases default so `--decoder viterbi` works out of the box.
  const calPath = join(cp.dest, "viterbi_calibration.json");
  const calUrl = `https://huggingface.co/${cp.repo}/resolve/main/viterbi_calibration.json`;
  const result = await downloadOne(calUrl, calPath, force, { optional: true });
  if (result === "missing" && (force || (await fileSize(calPath)) === null)) {
    await writeFile(calPath, JSON.stringify(DEFAULT_CALIBRATION, null, 2) + "\n");
    console.log(`        wrote ${rel(calPath)}  (zero-biases default)`);
  }
}

async function main() {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      model: { type: "string" },
      force: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
    strict: true,
  });

  if (values.help) {
    console.log(`Usage: bun scripts/prepare.ts [options]

Download model artifacts for privstrip from Hugging Face.

Options:
  --model <name>   Only fetch this checkpoint (default: all). Available:
${CHECKPOINTS.map((c) => `                     ${c.name.padEnd(14)} ${c.repo}`).join("\n")}
  --force          Re-download every file even when sizes already match.
  -h, --help

Files are skipped if their on-disk size matches the upstream content-length.
Each safetensors blob is ~2.8 GB; expect a one-time bandwidth cost of ~5.6 GB
for both checkpoints.
`);
    process.exit(0);
  }

  const filter = values.model as string | undefined;
  const selected = filter
    ? CHECKPOINTS.filter((c) => c.name === filter)
    : CHECKPOINTS;
  if (filter && selected.length === 0) {
    console.error(
      `unknown checkpoint: ${filter} ` +
        `(available: ${CHECKPOINTS.map((c) => c.name).join(", ")})`,
    );
    process.exit(2);
  }

  for (const cp of selected) {
    await prepareCheckpoint(cp, values.force as boolean);
  }

  console.log("\nready. Next:");
  console.log("  cargo build --release");
  console.log('  target/release/privstrip check -t "Call John at 555-1234"');
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
