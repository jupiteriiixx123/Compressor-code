#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def json_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float, bool)):
        return str(x)
    return json.dumps(x, ensure_ascii=False)


def serialize_history(session_ids, dates, sessions):
    if not (len(session_ids) == len(dates) == len(sessions)):
        raise ValueError(
            "haystack_session_ids, haystack_dates and haystack_sessions "
            "must have the same length"
        )

    blocks = []
    for idx, (session_id, date, turns) in enumerate(
        zip(session_ids, dates, sessions), start=1
    ):
        header = f"[Session {idx}"
        if date is not None and json_text(date):
            header += f" | {json_text(date)}"
        if session_id is not None and json_text(session_id):
            header += f" | id={json_text(session_id)}"
        header += "]"

        lines = [header]
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = json_text(turn.get("role")) or "unknown"
            content = json_text(turn.get("content"))
            # Deliberately ignore reward/evidence metadata such as has_answer.
            lines.append(f"{role.capitalize()}: {content}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks).strip()


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def percentile(sorted_values, p):
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="experience/raw/group1/longmemeval_s_cleaned.json",
    )
    ap.add_argument(
        "--output-dir",
        default="experience/processed/group1/longmemeval",
    )
    ap.add_argument(
        "--tokenizer",
        default="outputs/sft_full_600_v2_7ep",
    )
    ap.add_argument(
        "--context-limit",
        type=int,
        default=None,
        help="Optional compressor context limit used only for reporting; no truncation is performed.",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LongMemEval raw file must be a JSON list.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    policy_path = output_dir / "compressor_input.jsonl"
    meta_path = output_dir / "eval_meta.jsonl"

    policy_rows = []
    meta_rows = []

    for idx, sample in enumerate(data):
        question_id = str(sample.get("question_id", idx))
        item_id = f"longmemeval::{question_id}"

        session_ids = sample.get("haystack_session_ids", [])
        dates = sample.get("haystack_dates", [])
        sessions = sample.get("haystack_sessions", [])

        context = serialize_history(session_ids, dates, sessions)
        if not context:
            raise ValueError(f"{item_id}: empty serialized history")

        source_tokens = token_count(tokenizer, context)

        policy_rows.append({
            "id": item_id,
            "context": context,
            "source_tokens": source_tokens,
        })

        meta_rows.append({
            "id": item_id,
            "question_id": question_id,
            "question_type": sample.get("question_type"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "question_date": sample.get("question_date"),
            "answer_session_ids": sample.get("answer_session_ids", []),
            "haystack_session_ids": session_ids,
            "is_abstention": question_id.endswith("_abs"),
        })

    if [x["id"] for x in policy_rows] != [x["id"] for x in meta_rows]:
        raise RuntimeError("Policy/meta ID alignment failed.")

    with policy_path.open("w", encoding="utf-8") as f:
        for row in policy_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with meta_path.open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = sorted(x["source_tokens"] for x in policy_rows)
    print("LongMemEval preparation complete")
    print(f"samples        : {len(policy_rows)}")
    print(
        "source tokens  : "
        f"min={lengths[0]} "
        f"mean={sum(lengths)/len(lengths):.1f} "
        f"median={percentile(lengths, 0.50):.1f} "
        f"p90={percentile(lengths, 0.90):.1f} "
        f"p95={percentile(lengths, 0.95):.1f} "
        f"max={lengths[-1]}"
    )

    if args.context_limit is not None:
        over = sum(x > args.context_limit for x in lengths)
        print(f"> context limit: {over}/{len(lengths)} (limit={args.context_limit})")
        print("NOTE: prepare_longmemeval.py never truncates the benchmark.")

    print(f"policy output  : {policy_path}")
    print(f"eval output    : {meta_path}")


if __name__ == "__main__":
    main()
