#!/usr/bin/env bun
/**
 * Sample CloudWatch log groups and report PII detected by privstrip.
 *
 * Reads log groups + events through the AWS CLI (so it inherits your existing
 * AWS auth), pipes each event message into a single long-running `privstrip
 * stream` subprocess, and writes a per-group findings report.
 *
 * Defaults are conservative — you can scan up to a thousand groups by raising
 * --max-groups, and you can trade fidelity for speed by lowering --per-group.
 *
 * IMPORTANT: log events themselves may contain PII. By default the JSON report
 * stores only span labels + offsets + a redacted snippet (PII replaced with
 * <LABEL> placeholders). Pass --include-text to include the raw PII strings;
 * use only on a machine you're comfortable storing them on.
 */

import { spawn } from "bun";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";

const HERE = dirname(import.meta.path);
const PROJECT_ROOT = resolve(HERE, "..");

type EventRow = {
  logGroupName: string;
  logStreamName: string | null;
  timestamp: number | null;
  message: string;
};

type Span = {
  label: string;
  byte_start: number;
  byte_end: number;
  text: string;
};

type StreamReply = {
  id: string | number;
  spans?: Span[];
  tokens?: number;
  elapsed_us?: number;
  error?: string;
};

type GroupFinding = {
  events_scanned: number;
  events_with_pii: number;
  label_counts: Record<string, number>;
  examples: Array<{
    log_stream: string | null;
    timestamp: number | null;
    spans: Array<{ label: string; byte_start: number; byte_end: number; text?: string }>;
    redacted_message: string;
  }>;
};

type Args = {
  region?: string;
  profile?: string;
  maxGroups: number;
  perGroup: number;
  maxMessageBytes: number;
  filter?: string;
  startMinutesAgo: number;
  output: string;
  includeText: boolean;
  privstripBin: string;
  modelDir: string;
  metal: boolean;
  examplesPerGroup: number;
  awsConcurrency: number;
  queueDepth: number;
};

function parseCli(): Args {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      region: { type: "string" },
      profile: { type: "string" },
      "max-groups": { type: "string", default: "50" },
      "per-group": { type: "string", default: "25" },
      "max-message-bytes": { type: "string", default: "8192" },
      filter: { type: "string" },
      "start-minutes-ago": { type: "string", default: "1440" },
      output: { type: "string", default: "cloudwatch-pii-report.json" },
      "include-text": { type: "boolean", default: false },
      "privstrip-bin": {
        type: "string",
        default: join(PROJECT_ROOT, "target/release/privstrip"),
      },
      "model-dir": { type: "string", default: PROJECT_ROOT },
      metal: { type: "boolean", default: false },
      "examples-per-group": { type: "string", default: "5" },
      "aws-concurrency": { type: "string", default: "4" },
      "queue-depth": { type: "string", default: "64" },
      help: { type: "boolean", short: "h", default: false },
    },
    strict: true,
    allowPositionals: false,
  });

  if (values.help) {
    console.log(`Usage: bun scripts/scan-cloudwatch.ts [options]

Options:
  --region <r>                AWS region (defaults to AWS_REGION / aws config)
  --profile <p>               AWS named profile to use
  --max-groups <n>            Limit number of log groups scanned (default 50)
  --per-group <n>             Events sampled per group (default 25)
  --max-message-bytes <n>     Truncate event messages above this size (default 8192)
  --filter <substring>        Only scan groups whose name contains this substring
  --start-minutes-ago <n>     Look back this many minutes for events (default 1440)
  --output <file>             Where to write the JSON report (default ./cloudwatch-pii-report.json)
  --include-text              Include raw PII text in the report (default: redacted snippets only)
  --examples-per-group <n>    Number of example events kept per group (default 5)
  --privstrip-bin <path>      Path to the privstrip binary
  --model-dir <path>          Directory containing model.safetensors / config.json / tokenizer.json
  --metal                     Try to use the Apple Metal backend (default: CPU)
  --aws-concurrency <n>       Parallel AWS log-event fetches (default 4)
  --queue-depth <n>           Max events buffered between AWS fetches and inference (default 64)
  -h, --help                  Show this help
`);
    process.exit(0);
  }

  return {
    region: values.region,
    profile: values.profile,
    maxGroups: Number(values["max-groups"]),
    perGroup: Number(values["per-group"]),
    maxMessageBytes: Number(values["max-message-bytes"]),
    filter: values.filter,
    startMinutesAgo: Number(values["start-minutes-ago"]),
    output: values.output as string,
    includeText: values["include-text"] as boolean,
    privstripBin: values["privstrip-bin"] as string,
    modelDir: values["model-dir"] as string,
    metal: values.metal as boolean,
    examplesPerGroup: Number(values["examples-per-group"]),
    awsConcurrency: Number(values["aws-concurrency"]),
    queueDepth: Number(values["queue-depth"]),
  };
}

function awsArgs(args: Args, ...extra: string[]): string[] {
  const base = ["aws"];
  if (args.region) base.push("--region", args.region);
  if (args.profile) base.push("--profile", args.profile);
  base.push("--output", "json", ...extra);
  return base;
}

async function runJson<T>(cmd: string[]): Promise<T> {
  const proc = spawn({ cmd, stdout: "pipe", stderr: "pipe" });
  const [out, err, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  if (code !== 0) {
    throw new Error(`${cmd.join(" ")} exited ${code}: ${err.trim()}`);
  }
  return JSON.parse(out) as T;
}

async function listLogGroups(args: Args): Promise<string[]> {
  const groups: string[] = [];
  let nextToken: string | undefined;
  do {
    const cmd = awsArgs(args, "logs", "describe-log-groups");
    if (nextToken) cmd.push("--starting-token", nextToken);
    type Resp = {
      logGroups: { logGroupName: string }[];
      nextToken?: string;
    };
    const resp = await runJson<Resp>(cmd);
    for (const g of resp.logGroups ?? []) groups.push(g.logGroupName);
    nextToken = resp.nextToken;
    if (groups.length >= args.maxGroups * 2) break; // generous slack pre-filter
  } while (nextToken);

  let filtered = groups;
  if (args.filter) filtered = groups.filter((g) => g.includes(args.filter!));
  return filtered.slice(0, args.maxGroups);
}

async function fetchEvents(args: Args, group: string): Promise<EventRow[]> {
  const startMs = Date.now() - args.startMinutesAgo * 60_000;
  const cmd = awsArgs(
    args,
    "logs",
    "filter-log-events",
    "--log-group-name",
    group,
    "--limit",
    String(args.perGroup),
    "--start-time",
    String(startMs),
  );
  type Resp = {
    events: Array<{
      logStreamName?: string;
      timestamp?: number;
      message?: string;
    }>;
  };
  let resp: Resp;
  try {
    resp = await runJson<Resp>(cmd);
  } catch (e) {
    process.stderr.write(`  ! ${group}: ${(e as Error).message}\n`);
    return [];
  }
  return (resp.events ?? [])
    .filter((e) => typeof e.message === "string" && e.message.length > 0)
    .map((e) => ({
      logGroupName: group,
      logStreamName: e.logStreamName ?? null,
      timestamp: e.timestamp ?? null,
      message: e.message!.slice(0, args.maxMessageBytes),
    }));
}

type ProgressFrame = {
  scanned: number;
  withPii: number;
  estimatedTotal: number;
  groupsFetched: number;
  groupCount: number;
  queueDepth: number;
  startedAt: number;
};

class Progress {
  private isTty = Boolean(process.stderr.isTTY);
  private currentLineLen = 0;
  private lastNonTtyLine = 0;

  update(f: ProgressFrame): void {
    if (!this.isTty) {
      // Non-TTY: emit one line every 10 messages to avoid log spam.
      if (f.scanned - this.lastNonTtyLine < 10 && f.scanned !== f.estimatedTotal) return;
      this.lastNonTtyLine = f.scanned;
      process.stderr.write(this.format(f) + "\n");
      return;
    }
    this.clear();
    const line = this.format(f);
    process.stderr.write(line);
    this.currentLineLen = line.length;
  }

  note(line: string): void {
    if (!this.isTty) {
      process.stderr.write(line + "\n");
      return;
    }
    this.clear();
    process.stderr.write(line + "\n");
  }

  clear(): void {
    if (!this.isTty || this.currentLineLen === 0) return;
    process.stderr.write("\r" + " ".repeat(this.currentLineLen) + "\r");
    this.currentLineLen = 0;
  }

  private format(f: ProgressFrame): string {
    const elapsedMs = Date.now() - f.startedAt;
    const elapsedSec = elapsedMs / 1000;
    const rate = elapsedSec > 0 ? f.scanned / elapsedSec : 0;
    const remaining = Math.max(0, f.estimatedTotal - f.scanned);
    const etaSec = rate > 0 ? remaining / rate : 0;
    const pct = f.estimatedTotal > 0 ? Math.floor((f.scanned / f.estimatedTotal) * 100) : 0;
    return (
      `${pad(f.scanned, 5)}/${pad(f.estimatedTotal, 5)} (${pad(pct, 2)}%)` +
      ` groups ${pad(f.groupsFetched, 3)}/${pad(f.groupCount, 3)}` +
      ` queue ${pad(f.queueDepth, 3)}` +
      ` pii ${pad(f.withPii, 4)}` +
      ` ${rate.toFixed(1).padStart(5)} msg/s` +
      ` elapsed ${formatDuration(elapsedMs)}` +
      ` eta ${etaSec > 0 ? formatDuration(etaSec * 1000) : "--:--"}`
    );
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatDuration(ms: number): string {
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${pad(m, 2)}:${pad(s, 2)}`;
  return `${pad(m, 2)}:${pad(s, 2)}`;
}

function pad(n: number, width: number): string {
  return String(n).padStart(width, "0");
}

function placeholderFor(label: string): string {
  return `<${label.toUpperCase().replace(/[^A-Z0-9]+/g, "_").replace(/^_+|_+$/g, "")}>`;
}

function redactMessage(message: string, spans: Span[]): string {
  const sorted = [...spans].sort((a, b) => a.byte_start - b.byte_start);
  const bytes = Buffer.from(message, "utf8");
  let out = "";
  let cursor = 0;
  for (const s of sorted) {
    if (s.byte_start > cursor) out += bytes.subarray(cursor, s.byte_start).toString("utf8");
    out += placeholderFor(s.label);
    cursor = s.byte_end;
  }
  if (cursor < bytes.length) out += bytes.subarray(cursor).toString("utf8");
  return out;
}

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

  constructor(args: Args) {
    this.proc = spawn({
      cmd: [
        args.privstripBin,
        "stream",
        "-m",
        args.modelDir,
        ...(args.metal ? ["--metal"] : []),
      ],
      stdin: "pipe",
      stdout: "pipe",
      stderr: "inherit",
    });
    this.stdin = this.proc.stdin as unknown as Stdin;
    this.readerPromise = this.consumeStdout();
  }

  private async consumeStdout(): Promise<void> {
    const reader = (this.proc.stdout as ReadableStream<Uint8Array>).getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      this.buf += this.decoder.decode(value, { stream: true });
      let newlineIdx: number;
      while ((newlineIdx = this.buf.indexOf("\n")) !== -1) {
        const line = this.buf.slice(0, newlineIdx);
        this.buf = this.buf.slice(newlineIdx + 1);
        if (!line) continue;
        const cb = this.queue.shift();
        if (!cb) {
          process.stderr.write(`unexpected privstrip line: ${line}\n`);
          continue;
        }
        try {
          cb(JSON.parse(line) as StreamReply);
        } catch (e) {
          cb({ id: "<parse-error>", error: `parse: ${(e as Error).message}: ${line}` });
        }
      }
    }
    for (const cb of this.queue) cb({ id: "<eof>", error: "privstrip stream closed unexpectedly" });
    this.queue = [];
  }

  async detect(id: string | number, text: string): Promise<StreamReply> {
    const promise = new Promise<StreamReply>((resolve) => this.queue.push(resolve));
    this.stdin.write(JSON.stringify({ id, text }) + "\n");
    await this.stdin.flush?.();
    return promise;
  }

  async close(): Promise<void> {
    this.stdin.end();
    await this.readerPromise;
    await this.proc.exited;
  }
}

async function main() {
  const args = parseCli();
  const progress = new Progress();
  progress.note("listing log groups...");
  const groups = await listLogGroups(args);
  progress.note(
    `scanning ${groups.length} groups (region=${args.region ?? "default"}, ` +
      `aws-concurrency=${args.awsConcurrency}, queue-depth=${args.queueDepth})`,
  );

  const client = new StreamClient(args);
  const findings: Record<string, GroupFinding> = {};
  for (const g of groups) {
    findings[g] = {
      events_scanned: 0,
      events_with_pii: 0,
      label_counts: {},
      examples: [],
    };
  }

  type QueueItem = { group: string; eventIdx: number; ev: EventRow };
  const queue: QueueItem[] = [];
  let producerDone = false;

  let totalScanned = 0;
  let totalWithPii = 0;
  let totalTokens = 0;
  let totalInferenceUs = 0;
  let groupsFetched = 0;
  const startedAt = Date.now();

  const renderProgress = () => {
    // Estimate the remaining workload: groups fetched have known event counts;
    // unfetched groups conservatively assume the full per-group sample size.
    const knownEvents = Object.values(findings).reduce((acc, f) => acc + f.events_scanned, 0);
    const projected =
      knownEvents + (groups.length - groupsFetched) * args.perGroup;
    progress.update({
      scanned: totalScanned,
      withPii: totalWithPii,
      estimatedTotal: Math.max(projected, totalScanned),
      groupsFetched,
      groupCount: groups.length,
      queueDepth: queue.length,
      startedAt,
    });
  };

  const fetchAndEnqueue = async (group: string) => {
    const events = await fetchEvents(args, group);
    findings[group].events_scanned = events.length;
    groupsFetched += 1;
    for (let j = 0; j < events.length; j++) {
      // Backpressure: stop pushing while the queue is full.
      while (queue.length >= args.queueDepth) await sleep(5);
      queue.push({ group, eventIdx: j, ev: events[j] });
    }
  };

  const producer = async () => {
    let nextIdx = 0;
    const workers = Array.from({ length: args.awsConcurrency }, async () => {
      while (true) {
        const i = nextIdx++;
        if (i >= groups.length) return;
        try {
          await fetchAndEnqueue(groups[i]);
        } catch (e) {
          progress.note(`  ! ${groups[i]} fetch failed: ${(e as Error).message}`);
          groupsFetched += 1;
        }
      }
    });
    await Promise.all(workers);
    producerDone = true;
  };

  const consumer = async () => {
    while (!producerDone || queue.length > 0) {
      if (queue.length === 0) {
        await sleep(5);
        continue;
      }
      const item = queue.shift()!;
      const id = `${item.group}#${item.eventIdx}`;
      const reply = await client.detect(id, item.ev.message);
      totalScanned += 1;
      totalTokens += reply.tokens ?? 0;
      totalInferenceUs += reply.elapsed_us ?? 0;
      renderProgress();

      if (reply.error) {
        progress.note(`  ! ${item.group} #${item.eventIdx}: ${reply.error}`);
        continue;
      }
      const spans = reply.spans ?? [];
      if (spans.length === 0) continue;
      const f = findings[item.group];
      f.events_with_pii += 1;
      totalWithPii += 1;
      for (const s of spans) {
        f.label_counts[s.label] = (f.label_counts[s.label] ?? 0) + 1;
      }
      if (f.examples.length < args.examplesPerGroup) {
        f.examples.push({
          log_stream: item.ev.logStreamName,
          timestamp: item.ev.timestamp,
          spans: spans.map((s) => ({
            label: s.label,
            byte_start: s.byte_start,
            byte_end: s.byte_end,
            ...(args.includeText ? { text: s.text } : {}),
          })),
          redacted_message: redactMessage(item.ev.message, spans),
        });
      }
    }
  };

  await Promise.all([producer(), consumer()]);

  progress.clear();
  await client.close();

  const summary = {
    generated_at: new Date().toISOString(),
    region: args.region ?? null,
    args: {
      max_groups: args.maxGroups,
      per_group: args.perGroup,
      start_minutes_ago: args.startMinutesAgo,
      filter: args.filter ?? null,
      include_text: args.includeText,
    },
    totals: summarize(findings),
    by_group: findings,
  };

  const wallSeconds = (Date.now() - startedAt) / 1000;
  const inferenceSeconds = totalInferenceUs / 1_000_000;
  const perfSummary = {
    messages: totalScanned,
    tokens: totalTokens,
    wall_seconds: round(wallSeconds, 2),
    inference_seconds: round(inferenceSeconds, 2),
    tokens_per_second_inference: inferenceSeconds > 0 ? round(totalTokens / inferenceSeconds, 1) : 0,
    tokens_per_second_wall: wallSeconds > 0 ? round(totalTokens / wallSeconds, 1) : 0,
    messages_per_second_wall: wallSeconds > 0 ? round(totalScanned / wallSeconds, 2) : 0,
    mean_latency_ms_per_message:
      totalScanned > 0 ? round((totalInferenceUs / 1000) / totalScanned, 2) : 0,
  };
  (summary as unknown as Record<string, unknown>).performance = perfSummary;

  await mkdir(dirname(resolve(args.output)), { recursive: true });
  await writeFile(args.output, JSON.stringify(summary, null, 2));
  process.stderr.write(`\nwrote ${args.output}\n`);
  printConsoleSummary(findings);
  printPerfSummary(perfSummary);
}

function round(n: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(n * f) / f;
}

function printPerfSummary(p: ReturnType<typeof Object.assign> | Record<string, number>) {
  const r = p as Record<string, number>;
  process.stderr.write(
    `\nPerformance:\n` +
      `  ${r.messages} messages, ${r.tokens} tokens\n` +
      `  ${r.wall_seconds}s wall, ${r.inference_seconds}s inference\n` +
      `  ${r.tokens_per_second_inference} tok/s inference, ${r.tokens_per_second_wall} tok/s wall\n` +
      `  ${r.messages_per_second_wall} msg/s wall, ${r.mean_latency_ms_per_message} ms mean latency\n`,
  );
}

function summarize(findings: Record<string, GroupFinding>) {
  let scanned = 0;
  let withPii = 0;
  const labels: Record<string, number> = {};
  for (const f of Object.values(findings)) {
    scanned += f.events_scanned;
    withPii += f.events_with_pii;
    for (const [k, v] of Object.entries(f.label_counts)) {
      labels[k] = (labels[k] ?? 0) + v;
    }
  }
  return { events_scanned: scanned, events_with_pii: withPii, label_counts: labels };
}

function printConsoleSummary(findings: Record<string, GroupFinding>) {
  const rows = Object.entries(findings)
    .filter(([, f]) => f.events_with_pii > 0)
    .sort((a, b) => b[1].events_with_pii - a[1].events_with_pii);
  if (rows.length === 0) {
    process.stderr.write("no PII detected in sampled events\n");
    return;
  }
  process.stderr.write("\nGroups with PII (events_with_pii / events_scanned, label counts):\n");
  for (const [group, f] of rows) {
    const labels = Object.entries(f.label_counts)
      .map(([k, v]) => `${k}=${v}`)
      .join(" ");
    process.stderr.write(
      `  ${f.events_with_pii.toString().padStart(4)}/${f.events_scanned.toString().padEnd(4)}  ${group}  [${labels}]\n`,
    );
  }
}

await main();
