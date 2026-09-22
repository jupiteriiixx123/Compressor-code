#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


SESSION_RE = re.compile(r"^session_(\d+)$")


def json_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        return " ".join(json_text(v) for v in x if json_text(v))
    return json.dumps(x, ensure_ascii=False)


def serialize_turn(turn):
    speaker = json_text(turn.get("speaker")) or "Unknown"
    text = json_text(turn.get("text"))
    caption = json_text(turn.get("blip_caption"))

    parts = []
    if text:
        parts.append(text)
    if caption:
        parts.append(f"[Image: {caption}]")

    content = " ".join(parts).strip()
    return f"{speaker}: {content}" if content else f"{speaker}:"


def serialize_conversation(conversation):
    speaker_a = json_text(conversation.get("speaker_a"))
    speaker_b = json_text(conversation.get("speaker_b"))

    session_ids = []
    for key in conversation:
        m = SESSION_RE.match(key)
        if m:
            session_ids.append(int(m.group(1)))
    session_ids.sort()

    blocks = []
    if speaker_a or speaker_b:
        participants = ", ".join(x for x in [speaker_a, speaker_b] if x)
        blocks.append(f"[Participants]\n{participants}")

    for sid in session_ids:
        turns = conversation.get(f"session_{sid}", [])
        date_time = json_text(conversation.get(f"session_{sid}_date_time"))
        header = f"[Session {sid}"
        if date_time:
            header += f" | {date_time}"
        header += "]"

        lines = [header]
        for turn in turns:
            if isinstance(turn, dict):
                lines.append(serialize_turn(turn))
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks).strip()


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="experience/raw/group1/locomo10.json",
        help="Raw LoCoMo locomo10.json",
    )
    ap.add_argument(
        "--output-dir",
        default="experience/processed/group1/locomo",
    )
    ap.add_argument(
        "--tokenizer",
        default="outputs/sft_full_600_v2_7ep",
        help="Tokenizer used by the compressor.",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LoCoMo raw file must be a JSON list.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    policy_path = output_dir / "compressor_input.jsonl"
    meta_path = output_dir / "eval_meta.jsonl"

    policy_rows = []
    meta_rows = []

    for sample_idx, sample in enumerate(data):
        sample_id = str(sample.get("sample_id", sample_idx))
        item_id = f"locomo::{sample_id}"

        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            raise ValueError(f"{item_id}: missing/invalid conversation")

        context = serialize_conversation(conversation)
        if not context:
            raise ValueError(f"{item_id}: empty serialized context")

        source_tokens = token_count(tokenizer, context)

        policy_rows.append({
            "id": item_id,
            "context": context,
            "source_tokens": source_tokens,
        })

        qas = []
        for q_idx, qa in enumerate(sample.get("qa", [])):
            if not isinstance(qa, dict):
                continue

            # Preserve both answer and adversarial_answer.  Category-5 handling
            # is intentionally deferred to the evaluator.
            qas.append({
                "question_id": f"{item_id}::q{q_idx:04d}",
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "adversarial_answer": qa.get("adversarial_answer"),
                "category": qa.get("category"),
                "evidence": qa.get("evidence", []),
            })

        meta_rows.append({
            "id": item_id,
            "sample_id": sample_id,
            "questions": qas,
        })

    if [x["id"] for x in policy_rows] != [x["id"] for x in meta_rows]:
        raise RuntimeError("Policy/meta ID alignment failed.")

    with policy_path.open("w", encoding="utf-8") as f:
        for row in policy_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with meta_path.open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [x["source_tokens"] for x in policy_rows]
    n_questions = sum(len(x["questions"]) for x in meta_rows)

    print("LoCoMo preparation complete")
    print(f"samples        : {len(policy_rows)}")
    print(f"questions      : {n_questions}")
    print(f"source tokens  : min={min(lengths)} mean={sum(lengths)/len(lengths):.1f} max={max(lengths)}")
    print(f"policy output  : {policy_path}")
    print(f"eval output    : {meta_path}")


if __name__ == "__main__":
    main()
