"""Run the official openai/privacy-filter on test inputs and dump per-token labels.

Usage:
    python run_reference.py "text to scan"
    python run_reference.py < input.txt

Loads the local checkpoint from ../ (which holds config.json, tokenizer.json,
model.safetensors) so this script doesn't redownload the 2.8 GB weights.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

from opf import load_model

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = sys.stdin.read()

    runtime = load_model(str(MODEL_DIR))
    spans = runtime.predict_text(text)

    out = {
        "text": text,
        "spans": [
            {
                "label": s.label,
                "char_start": s.start,
                "char_end": s.end,
                "text": s.text,
            }
            for s in spans
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
