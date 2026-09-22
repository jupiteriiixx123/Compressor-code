#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DATASETS = ("gov_report", "multi_news")


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
    return rows


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def prepare_dataset(name, input_path, output_root, tokenizer):
    rows = read_jsonl(input_path)
    out_dir = Path(output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_path = out_dir / "compressor_input.jsonl"
    meta_path = out_dir / "eval_meta.jsonl"

    policy_rows = []
    meta_rows = []

    for idx, sample in enumerate(rows):
        raw_id = sample.get("_id", idx)
        item_id = f"{name}::{raw_id}"

        context = sample.get("context")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"{item_id}: missing/empty context")
        context = context.strip()

        source_tokens = token_count(tokenizer, context)

        policy_rows.append({
            "id": item_id,
            "context": context,
            "source_tokens": source_tokens,
        })

        # Keep benchmark-side task/gold metadata entirely outside the compressor file.
        meta_rows.append({
            "id": item_id,
            "_id": raw_id,
            "dataset": sample.get("dataset", name),
            "input": sample.get("input"),
            "gold_summaries": sample.get("answers", []),
            "length": sample.get("length"),
            "language": sample.get("language"),
            "all_classes": sample.get("all_classes"),
        })

    if [x["id"] for x in policy_rows] != [x["id"] for x in meta_rows]:
        raise RuntimeError(f"{name}: Policy/meta ID alignment failed.")

    with policy_path.open("w", encoding="utf-8") as f:
        for row in policy_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with meta_path.open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [x["source_tokens"] for x in policy_rows]
    print(f"{name} preparation complete")
    print(f"samples        : {len(policy_rows)}")
    print(f"source tokens  : min={min(lengths)} mean={sum(lengths)/len(lengths):.1f} max={max(lengths)}")
    print(f"policy output  : {policy_path}")
    print(f"eval output    : {meta_path}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir",
        default="experience/raw/group2",
        help="Directory containing gov_report.jsonl and multi_news.jsonl",
    )
    ap.add_argument(
        "--output-root",
        default="experience/processed/group2",
    )
    ap.add_argument(
        "--tokenizer",
        default="outputs/sft_full_600_v2_7ep",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    raw_dir = Path(args.raw_dir)
    for name in DATASETS:
        prepare_dataset(
            name=name,
            input_path=raw_dir / f"{name}.jsonl",
            output_root=args.output_root,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
