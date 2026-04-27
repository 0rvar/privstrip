# python-ref

Reference implementation wrapper around the official
[openai/privacy-filter](https://github.com/openai/privacy-filter) (`opf`) Python
package. Used as the third leg of the three-way validation matrix described in
`../VALIDATION.md`.

The HF checkpoint and the `opf` GitHub package were uploaded a few days apart
with mismatched config schemas (different field names, missing keys). This
wrapper translates `models/config.json` into the schema `opf` expects, drops the
result into a side-car directory next to a symlink of `model.safetensors`, and
points `opf` at the side-car. Weights are not modified.

## Setup

```fish
nix develop                       # gets you uv + python311 + cargo + bun
cd python-ref
uv sync                           # installs opf + torch CPU + tiktoken
```

CPU is the only supported device on Apple Silicon; `opf` only accepts `cpu` /
`cuda` literals.

## Stream protocol

Identical to the Rust binary's `stream` mode:

```fish
echo '{"id":1,"text":"Call John at 555-1234"}' \
  | uv run python run_reference.py stream --decoder argmax
```

Replies are JSONL with shape `{id, spans:[{label,byte_start,byte_end,text}], tokens, elapsed_us}`.
Errors come back as `{id, error}`.

## Debug mode

```fish
uv run python run_reference.py debug --decoder argmax -t "Call John at 555-1234"
```

Emits per-token rows similar to `privstrip debug`. Useful when triaging
Cluster C (model output) divergences.

## Caveats

- `opf` returns char offsets; we convert to UTF-8 byte offsets so the
  comparator can diff against the Rust port.
- When the tokenizer round-trip produces a different string from the input
  (`decoded_mismatch=true`), `opf` shifts spans onto the decoded form. The
  reply payload includes a `decoded_mismatch` flag so the comparator can skip
  the row.
