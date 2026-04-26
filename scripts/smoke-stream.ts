#!/usr/bin/env bun
/**
 * Local smoke test for the long-running privstrip stream subprocess.
 * Verifies that the duplex stdin/stdout pump in scan-cloudwatch.ts works.
 */

import { spawn } from "bun";
import { dirname, join, resolve } from "node:path";

const HERE = dirname(import.meta.path);
const PROJECT_ROOT = resolve(HERE, "..");
const BIN = process.env.PRIVSTRIP_BIN ?? join(PROJECT_ROOT, "target/release/privstrip");
const MODEL_DIR = process.env.PRIVSTRIP_MODEL_DIR ?? PROJECT_ROOT;

const events = [
  { id: "a", text: "Hello, my name is Quindle Testwick and my email is quindle@example.com." },
  { id: 2, text: "Contact Alice Smith at alice@gmail.com or call (555) 123-4567." },
  { id: "c", text: "John Smith lives at 123 Main Street and his SSN is 123-45-6789." },
  { id: "clean", text: "this string contains no personal data" },
  { id: "empty", text: "" },
];

const proc = spawn({
  cmd: [BIN, "stream", "-m", MODEL_DIR],
  stdin: "pipe",
  stdout: "pipe",
  stderr: "inherit",
});

const stdin = proc.stdin as { write: (s: string) => number; end: () => void; flush?: () => void | Promise<void> };
const reader = (proc.stdout as ReadableStream<Uint8Array>).getReader();
const decoder = new TextDecoder();
let buf = "";

const expected = events.length;
let received = 0;

const readLoop = (async () => {
  while (received < expected) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 1);
      if (!line) continue;
      const reply = JSON.parse(line);
      console.log(JSON.stringify(reply));
      received += 1;
    }
  }
})();

for (const ev of events) {
  stdin.write(JSON.stringify(ev) + "\n");
  await stdin.flush?.();
}
stdin.end();
await readLoop;
await proc.exited;

if (received !== expected) {
  console.error(`expected ${expected} replies, got ${received}`);
  process.exit(1);
}
console.error("ok");
