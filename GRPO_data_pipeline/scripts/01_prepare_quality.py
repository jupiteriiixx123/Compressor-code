#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_data.io_utils import write_jsonl
from stage2_data.quality import prepare_quality


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--split", required=True, choices=["train", "dev", "test"])
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", default=None)
    args = p.parse_args()

    rows = prepare_quality(args.input, args.split)

    if args.tokenizer:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for row in rows:
            row["source_tokens"] = len(tok.encode(row["context"], add_special_tokens=False))

    write_jsonl(args.output, rows)

    probe_counts = [len(x["probes"]) for x in rows]
    print(f"articles={len(rows)}")
    print(f"probes={sum(probe_counts)}")
    if probe_counts:
        print(
            f"probes/article: min={min(probe_counts)} "
            f"mean={sum(probe_counts)/len(probe_counts):.2f} max={max(probe_counts)}"
        )
    print(f"written={args.output}")


if __name__ == "__main__":
    main()
