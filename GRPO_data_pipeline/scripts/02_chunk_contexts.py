#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_data.chunking import chunk_text
from stage2_data.io_utils import read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--chunk-tokens", type=int, default=1600)
    p.add_argument("--overlap-tokens", type=int, default=150)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out, per_article = [], []

    for article in read_jsonl(args.input):
        chunks = chunk_text(
            article["context"],
            tok,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
        )
        per_article.append(len(chunks))
        for chunk in chunks:
            out.append({
                "id": f"{article['id']}::{chunk['chunk_id']}",
                "context_id": article["id"],
                "article_id": article.get("article_id"),
                **chunk,
            })

    write_jsonl(args.output, out)
    print(f"articles={len(per_article)} chunks={len(out)}")
    if per_article:
        print(
            f"chunks/article: min={min(per_article)} "
            f"mean={sum(per_article)/len(per_article):.2f} max={max(per_article)}"
        )
    print(f"written={args.output}")


if __name__ == "__main__":
    main()
