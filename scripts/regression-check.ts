#!/usr/bin/env bun
/**
 * Pre-merge regression check.
 *
 * Builds the release binary, runs the validation matrix against the cached
 * Python results, and fails the run if either A↔C pair drops below the
 * production threshold. The python cache lives in scripts/.python-cache.jsonl
 * and is treated as the reference; this script never invokes Python.
 *
 * Exit codes:
 *   0  matrix held — both thresholds met
 *   1  threshold violation
 *   2  internal error (build failure, missing files, etc.)
 *
 * CLAUDE.md: "After every change, run bun scripts/validate.ts ... and confirm
 * A_argmax_vs_C_argmax >= 96.80% and A_viterbi_vs_C_viterbi >= 99.00%."
 */

import { spawn } from "bun";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { existsSync } from "node:fs";
import { parseArgs } from "node:util";

const HERE = dirname(import.meta.path);
const PROJECT_ROOT = resolve(HERE, "..");

const THRESHOLDS = {
  A_argmax_vs_C_argmax: 96.80,
  A_viterbi_vs_C_viterbi: 99.00,
};

type Pair = keyof typeof THRESHOLDS;

async function run(cmd: string[], opts: { cwd?: string; description?: string } = {}): Promise<void> {
  process.stderr.write(`+ ${cmd.join(" ")}\n`);
  const proc = spawn({ cmd, stdin: "inherit", stdout: "inherit", stderr: "inherit", cwd: opts.cwd });
  const code = await proc.exited;
  if (code !== 0) {
    throw new Error(`${opts.description ?? cmd[0]} exited with code ${code}`);
  }
}

async function main() {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      "skip-build": { type: "boolean", default: false },
      "max-rows": { type: "string", default: "500" },
      help: { type: "boolean", short: "h", default: false },
    },
    strict: true,
  });

  if (values.help) {
    console.log(`Usage: bun scripts/regression-check.ts [options]

Options:
  --skip-build         Don't run cargo build --release (use existing binary)
  --max-rows <n>       Corpus rows to compare (default 500)
  -h, --help           Show this message
`);
    process.exit(0);
  }

  if (!values["skip-build"]) {
    await run(["cargo", "build", "--release"], {
      cwd: PROJECT_ROOT,
      description: "cargo build --release",
    });
  }

  const cachePath = join(PROJECT_ROOT, "scripts/.python-cache.jsonl");
  if (!existsSync(cachePath)) {
    console.error(
      `error: ${cachePath} not found. Populate the python cache first:\n` +
        `  cd python-ref && uv sync && cd ..\n` +
        `  bun scripts/validate.ts --max-rows 500   # auto-populates the cache`,
    );
    process.exit(2);
  }

  const matrixPath = join(PROJECT_ROOT, "validation-matrix.json");
  await run(
    [
      "bun",
      "scripts/validate.ts",
      "--no-python",
      "--max-rows",
      String(values["max-rows"]),
      "--matrix-out",
      matrixPath,
    ],
    {
      cwd: PROJECT_ROOT,
      description: "validate.ts",
    },
  );

  const matrix = JSON.parse(await readFile(matrixPath, "utf-8")) as Record<
    Pair,
    { exact_pct: number; total: number; mismatched: number }
  >;

  let failed = false;
  console.log("");
  console.log("=== regression check ===");
  for (const [pair, threshold] of Object.entries(THRESHOLDS) as Array<[Pair, number]>) {
    const stats = matrix[pair];
    if (!stats) {
      console.log(`  ${pair}: MISSING from matrix`);
      failed = true;
      continue;
    }
    const ok = stats.exact_pct >= threshold;
    const tag = ok ? "OK " : "FAIL";
    const pad = pair.padEnd(28);
    console.log(
      `  [${tag}] ${pad} ${stats.exact_pct.toFixed(2)}% >= ${threshold.toFixed(2)}%  (${stats.mismatched} mismatched / ${stats.total})`,
    );
    if (!ok) failed = true;
  }

  if (failed) {
    console.log("");
    console.log("regression check FAILED — at least one A↔C pair dropped below threshold");
    process.exit(1);
  }
  console.log("");
  console.log("regression check OK");
}

main().catch((e) => {
  console.error("regression-check error:", e);
  process.exit(2);
});
