#!/usr/bin/env python3
"""
Dataset-agnostic QA reader for downstream QA evaluation.

Use cases:
- compressed memory: Stage-0 / Stage-1 / Stage-2 compressor outputs
- full-context baseline: processed compressor_input.jsonl

The reader prompt is FIXED in this file.
The reader NEVER receives gold answers.
No input truncation is performed.
"""
# prompt
"""
SYSTEM:
Answer the question using only the provided context. Be concise and do not add unsupported information.

USER:
CONTEXT:
<memory>

QUESTION DATE:
<optional>

QUESTION:
<question>
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed


# ---------------------------------------------------------------------
# FIXED READER PROMPT
# Keep identical across full-context / Stage-0 / Stage-1 / Stage-2.
# ---------------------------------------------------------------------
READER_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Be concise and do not add unsupported information."
)


# LongMemEval full-context-only configuration.
LONG_CONTEXT_MAX_TOKENS = 131072
LONG_CONTEXT_YARN_FACTOR = 4.0
LONG_CONTEXT_ORIGINAL_MAX_POSITION = 32768


def parse_args():
    p = argparse.ArgumentParser()

    # Inputs / outputs
    p.add_argument("--memory-input", required=True,
                   help="Compression JSONL or processed compressor_input.jsonl.")
    p.add_argument("--eval-meta", required=True,
                   help="Processed eval_meta.jsonl.")
    p.add_argument("--output", required=True,
                   help="Prediction JSONL.")
    p.add_argument(
        "--memory-mode",
        choices=["compressed", "full_context"],
        default="compressed",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-sources", type=int, default=None)

    # Reader model
    p.add_argument("--reader-model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--trust-remote-code", action="store_true", default=True)

    # Qwen3 mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--thinking", dest="thinking", action="store_true")
    mode.add_argument("--no-thinking", dest="thinking", action="store_false")
    p.set_defaults(thinking=False)

    # Generation
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    # Audit guard only; NEVER truncates
    p.add_argument("--context-limit", type=int, default=None)
    p.add_argument(
        "--long-context",
        action="store_true",
        help="Enable 131K YaRN + FlashAttention-2 for LongMemEval full-context reading.",
    )

    return p.parse_args()


def torch_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
            if "id" not in row:
                raise ValueError(f"{path}:{line_no}: missing id")
            rows.append(row)
    return rows


def index_by_id(rows: List[Dict[str, Any]], name: str):
    out = {}
    for row in rows:
        sid = str(row["id"])
        if sid in out:
            raise ValueError(f"Duplicate id in {name}: {sid}")
        out[sid] = row
    return out


def flatten_compression(row: Dict[str, Any]) -> str:
    """
    Same eager-flattened user-facing view used by the current experiments.
    """
    obj = row.get("output")

    if isinstance(obj, dict):
        summary = obj.get("summary")
        link = obj.get("link")

        if isinstance(summary, str) and isinstance(link, dict):
            parts = [f"Summary:\n{summary}"]

            if link:
                detail_lines = []
                for key, value in link.items():
                    if isinstance(key, str) and isinstance(value, str):
                        detail_lines.append(f"[{key}] {value}")

                if detail_lines:
                    parts.append(
                        "Expandable information:\n" + "\n".join(detail_lines)
                    )

            return "\n\n".join(parts)

    raw = row.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        return raw

    raise ValueError(f"{row.get('id')}: no usable compression content")


def get_memory(row: Dict[str, Any], mode: str) -> str:
    if mode == "compressed":
        return flatten_compression(row)

    context = row.get("context")
    if not isinstance(context, str) or not context:
        raise ValueError(f"{row.get('id')}: full_context row has no context")
    return context


def normalize_questions(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Support both:
      A) one source -> many questions  (LoCoMo-style)
      B) one source -> one question    (LongMemEval-style)

    Gold answers are intentionally ignored here.
    """
    source_id = str(meta["id"])

    if isinstance(meta.get("questions"), list):
        out = []
        for i, q in enumerate(meta["questions"]):
            if not isinstance(q, dict):
                continue

            question = q.get("question")
            if not isinstance(question, str) or not question.strip():
                continue

            out.append({
                "question_id": str(
                    q.get("question_id", f"{source_id}::q{i:04d}")
                ),
                "question": question,
                "question_date": q.get("question_date"),
            })

        return out

    question = meta.get("question")
    if isinstance(question, str) and question.strip():
        return [{
            "question_id": str(
                meta.get("question_id", f"{source_id}::q0000")
            ),
            "question": question,
            "question_date": meta.get("question_date"),
        }]

    raise ValueError(f"{source_id}: no usable question(s)")


def build_user_prompt(
    memory: str,
    question: str,
    question_date: Optional[str],
) -> str:
    parts = [
        "CONTEXT:",
        memory,
        "",
    ]

    if question_date:
        parts.extend([
            "QUESTION DATE:",
            str(question_date),
            "",
        ])

    parts.extend([
        "QUESTION:",
        question,
    ])

    return "\n".join(parts)


def generation_config(args):
    # Qwen3 recommended sampling defaults.
    if args.thinking:
        default_t, default_p, default_k = 0.6, 0.95, 20
    else:
        default_t, default_p, default_k = 0.7, 0.8, 20

    temperature = default_t if args.temperature is None else args.temperature
    top_p = default_p if args.top_p is None else args.top_p
    top_k = default_k if args.top_k is None else args.top_k

    if temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if not (0 < top_p <= 1):
        raise ValueError("--top-p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if not (0 <= args.min_p <= 1):
        raise ValueError("--min-p must be in [0, 1]")

    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": args.min_p,
        "num_return_sequences": 1,
        "max_new_tokens": args.max_new_tokens,
    }


def stable_question_seed(
    global_seed: int,
    source_id: str,
    question_id: str,
) -> int:
    key = f"{source_id}\n{question_id}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return (global_seed + int.from_bytes(h[:8], "big")) % (2**31 - 1)


def seed_everything(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reader(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.reader_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(
        args.reader_model,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs = dict(
        config=config,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if args.long_context:
        # Intended for the LongMemEval FULL-CONTEXT baseline only.
        # Compressed-memory reader runs should normally omit --long-context.
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": LONG_CONTEXT_YARN_FACTOR,
            "original_max_position_embeddings": LONG_CONTEXT_ORIGINAL_MAX_POSITION,
        }
        config.max_position_embeddings = LONG_CONTEXT_MAX_TOKENS
        # model_kwargs["attn_implementation"] = "flash_attention_2"
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.reader_model,
        **model_kwargs,
    )

    model.to(args.device)
    model.eval()
    return model, tokenizer


def effective_context_limit(args) -> Optional[int]:
    if args.long_context:
        if args.context_limit is None:
            return LONG_CONTEXT_MAX_TOKENS
        return min(args.context_limit, LONG_CONTEXT_MAX_TOKENS)
    return args.context_limit


def build_inputs(
    tokenizer,
    memory: str,
    question: str,
    question_date: Optional[str],
    thinking: bool,
):
    user_prompt = build_user_prompt(
        memory=memory,
        question=question,
        question_date=question_date,
    )

    messages = [
        {"role": "system", "content": READER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = dict(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    try:
        return tokenizer.apply_chat_template(
            **kwargs,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(**kwargs)


def extract_final_answer(text: str) -> str:
    stripped = text.strip()
    if "</think>" in stripped:
        stripped = stripped.rsplit("</think>", 1)[-1].strip()
    return stripped


@torch.inference_mode()
def answer_one(
    model,
    tokenizer,
    memory: str,
    question: str,
    question_date: Optional[str],
    thinking: bool,
    device: str,
    generation_kwargs: Dict[str, Any],
    context_limit: Optional[int],
):
    inputs = build_inputs(
        tokenizer=tokenizer,
        memory=memory,
        question=question,
        question_date=question_date,
        thinking=thinking,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    if (
        context_limit is not None
        and prompt_tokens + generation_kwargs["max_new_tokens"] > context_limit
    ):
        raise RuntimeError(
            f"Reader prompt has {prompt_tokens} tokens and max_new_tokens="
            f"{generation_kwargs['max_new_tokens']}; total exceeds "
            f"--context-limit={context_limit}. Input is NOT truncated."
        )

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids = outputs[0, prompt_tokens:]
    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    prediction = extract_final_answer(raw_output)

    return (
        prediction,
        raw_output,
        prompt_tokens,
        int(generated_ids.shape[-1]),
    )


def completed_keys(output_path: Path):
    done = set()

    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Existing output is invalid JSONL at "
                    f"{output_path}:{line_no}"
                ) from e

            sid = row.get("source_id")
            qid = row.get("question_id")

            if sid is not None and qid is not None:
                done.add((str(sid), str(qid)))

    return done


def main():
    args = parse_args()

    memory_path = Path(args.memory_input)
    meta_path = Path(args.eval_meta)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    memory_rows = read_jsonl(memory_path)
    meta_rows = read_jsonl(meta_path)

    if args.max_sources is not None:
        memory_rows = memory_rows[:args.max_sources]

    memory_index = index_by_id(memory_rows, "memory input")
    meta_index = index_by_id(meta_rows, "eval metadata")

    source_ids = list(memory_index.keys())

    missing_meta = [sid for sid in source_ids if sid not in meta_index]
    if missing_meta:
        raise ValueError(
            f"{len(missing_meta)} memory rows have no eval metadata. "
            f"First: {missing_meta[:3]}"
        )

    jobs = []
    for source_id in source_ids:
        for q in normalize_questions(meta_index[source_id]):
            jobs.append((source_id, q))

    done = completed_keys(output_path)
    pending = [
        (source_id, q)
        for source_id, q in jobs
        if (source_id, q["question_id"]) not in done
    ]

    gen_cfg = generation_config(args)
    context_limit = effective_context_limit(args)

    print("=" * 72)
    print("DATASET-AGNOSTIC QA READER")
    print("=" * 72)
    print(f"memory input      : {memory_path}")
    print(f"eval metadata     : {meta_path}")
    print(f"output            : {output_path}")
    print(f"memory mode       : {args.memory_mode}")
    print(f"reader model      : {args.reader_model}")
    print(f"device / dtype    : {args.device} / {args.dtype}")
    print(f"thinking          : {args.thinking}")
    print(f"seed              : {args.seed}")
    print(
        "sampling           : "
        f"T={gen_cfg['temperature']} "
        f"top_p={gen_cfg['top_p']} "
        f"top_k={gen_cfg['top_k']} "
        f"min_p={gen_cfg['min_p']}"
    )
    print(f"max_new_tokens    : {gen_cfg['max_new_tokens']}")
    print(f"long context      : {args.long_context}")
    if args.long_context:
        print(f"YaRN               : factor={LONG_CONTEXT_YARN_FACTOR}, original={LONG_CONTEXT_ORIGINAL_MAX_POSITION}")
        print("attention          : flash_attention_2")
    print(f"context limit     : {context_limit if context_limit is not None else '[not explicitly set]'}")
    print(f"sources           : {len(source_ids)}")
    print(f"questions total   : {len(jobs)}")
    print(f"already completed : {len(jobs) - len(pending)}")
    print(f"pending           : {len(pending)}")
    print("reader prompt     : FIXED")
    print("gold answer input : NEVER")
    print("input truncation  : NEVER")
    print("=" * 72)

    if not pending:
        print("Nothing to do.")
        return

    model, tokenizer = load_reader(args)

    mode = "a" if output_path.exists() else "w"
    memory_cache: Dict[str, str] = {}

    with output_path.open(mode, encoding="utf-8") as fout:
        for source_id, q in tqdm(pending, desc="Reading"):
            if source_id not in memory_cache:
                memory_cache[source_id] = get_memory(
                    memory_index[source_id],
                    args.memory_mode,
                )

            memory = memory_cache[source_id]
            question_id = q["question_id"]
            question = q["question"]
            question_date = q.get("question_date")

            sseed = stable_question_seed(
                args.seed,
                source_id,
                question_id,
            )
            seed_everything(sseed)

            (
                prediction,
                raw_output,
                prompt_tokens,
                generated_tokens,
            ) = answer_one(
                model=model,
                tokenizer=tokenizer,
                memory=memory,
                question=question,
                question_date=question_date,
                thinking=args.thinking,
                device=args.device,
                generation_kwargs=gen_cfg,
                context_limit=context_limit,
            )

            result = {
                "source_id": source_id,
                "question_id": question_id,
                "prediction": prediction,
                "raw_output": raw_output,
                "question": question,
                "question_date": question_date,
                "memory_mode": args.memory_mode,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "sample_seed": sseed,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()