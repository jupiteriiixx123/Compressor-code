#!/usr/bin/env python3
"""
Dataset-agnostic benchmark compressor runner.

Input JSONL schema:
    {"id": "...", "context": "...", "source_tokens": 12345}

Important:
- The compressor prompt is FIXED in this file.
- It matches the prompt protocol used in Stage-1 SFT / Stage-2 RL:
    SYSTEM = fixed compressor instruction
    USER   = raw context only
- The script never reads eval_meta.jsonl.
- No input truncation is performed.
"""
# prompt
"""
SYSTEM:
You are a long-context compression model. Compress the document into a coherent summary with expandable semantic information blocks. Output valid JSON with exactly two keys: "summary" and "link". Use ASCII citation anchors in the form [cite_n]. The summary length, citation count, block count, block length, and compression ratio must adapt freely to the document's information content.

USER:
<context>
"""


import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# ---------------------------------------------------------------------
# FIXED TRAINING-TIME COMPRESSOR PROMPT
# Keep unchanged for Stage-0 / Stage-1 / Stage-2 benchmark inference.
# ---------------------------------------------------------------------
COMPRESSOR_SYSTEM_PROMPT = (
    "You are a long-context compression model. Compress the document into a coherent summary "
    "with expandable semantic information blocks. Output valid JSON with exactly two keys: "
    "\"summary\" and \"link\". Use ASCII citation anchors in the form [cite_n]. The summary length, "
    "citation count, block count, block length, and compression ratio must adapt freely to the "
    "document's information content."
)


# LongMemEval-only long-context configuration.
LONG_CONTEXT_MAX_TOKENS = 131072
LONG_CONTEXT_YARN_FACTOR = 4.0
LONG_CONTEXT_ORIGINAL_MAX_POSITION = 32768


def parse_args():
    p = argparse.ArgumentParser()

    # Data I/O
    p.add_argument("--input", required=True,
                   help="Processed compressor_input.jsonl")
    p.add_argument("--output", required=True,
                   help="Output compression JSONL")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output instead of resuming.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Optional smoke-test limit.")

    # Model
    p.add_argument("--base-model", required=True,
                   help="Base/full checkpoint. Stage-0: original Qwen; Stage-1/2: SFT checkpoint.")
    p.add_argument("--adapter", default=None,
                   help="Optional Stage-2 LoRA adapter.")
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
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)

    # Optional audit guard only; NEVER truncates
    p.add_argument("--context-limit", type=int, default=None)

    p.add_argument(
    "--long-context",
    action="store_true",
    help="Enable YaRN rope scaling for very long inputs (used for LongMemEval only).",
    )

    return p.parse_args()


def torch_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


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


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs = dict(
        config=config,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if args.long_context:
        # Static YaRN used only for LongMemEval. Other benchmarks keep the
        # checkpoint's original RoPE/configuration unchanged.
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": LONG_CONTEXT_YARN_FACTOR,
            "original_max_position_embeddings": LONG_CONTEXT_ORIGINAL_MAX_POSITION,
        }
        config.max_position_embeddings = LONG_CONTEXT_MAX_TOKENS
        # model_kwargs["attn_implementation"] = "flash_attention_2"
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    if args.adapter:
        if PeftModel is None:
            raise RuntimeError("peft is required when --adapter is provided.")
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            is_trainable=False,
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


def read_jsonl(path: Path):
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

            missing = {"id", "context", "source_tokens"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {sorted(missing)}")

            if not isinstance(row["context"], str) or not row["context"]:
                raise ValueError(f"{path}:{line_no}: empty context")

            rows.append(row)
    return rows


def existing_ids(path: Path):
    if not path.exists():
        return set()

    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line)["id"]))
            except Exception as e:
                raise RuntimeError(
                    f"Invalid existing output at {path}:{line_no}"
                ) from e
    return ids


def sample_seed(global_seed: int, sample_id: str) -> int:
    h = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (global_seed + int.from_bytes(h[:8], "big")) % (2**31 - 1)


def seed_everything(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_inputs(tokenizer, context: str, thinking: bool):
    # EXACT inference protocol:
    #   system -> fixed training prompt
    #   user   -> raw context only
    messages = [
        {"role": "system", "content": COMPRESSOR_SYSTEM_PROMPT},
        {"role": "user", "content": context},
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


def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    stripped = text.strip()

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, "exact"
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, "extracted"
        except json.JSONDecodeError:
            pass

    return None, None


def validate_structure(obj: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"summary", "link"}:
        return False

    summary = obj["summary"]
    link = obj["link"]

    if not isinstance(summary, str) or not summary.strip():
        return False
    if not isinstance(link, dict):
        return False

    import re

    summary_cites = re.findall(r"\[(cite_\d+)\]", summary)
    summary_set = set(summary_cites)
    link_set = set(link.keys())

    if summary_set != link_set:
        return False
    if len(summary_cites) != len(summary_set):
        return False

    for key, value in link.items():
        if re.fullmatch(r"cite_\d+", key) is None:
            return False
        if not isinstance(value, str) or not value.strip():
            return False

    return True


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    context: str,
    device: str,
    thinking: bool,
    generation_kwargs: Dict[str, Any],
    context_limit: Optional[int],
):
    inputs = build_inputs(tokenizer, context, thinking)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    if (
        context_limit is not None
        and prompt_tokens + generation_kwargs["max_new_tokens"] > context_limit
    ):
        raise RuntimeError(
            f"Prompt has {prompt_tokens} tokens and max_new_tokens="
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

    return raw_output, prompt_tokens, int(generated_ids.shape[-1])


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    rows = read_jsonl(input_path)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    done = existing_ids(output_path)
    pending = [row for row in rows if str(row["id"]) not in done]

    gen_cfg = generation_config(args)
    context_limit = effective_context_limit(args)

    print("=" * 72)
    print("DATASET-AGNOSTIC COMPRESSOR")
    print("=" * 72)
    print(f"input             : {input_path}")
    print(f"output            : {output_path}")
    print(f"base model        : {args.base_model}")
    print(f"adapter           : {args.adapter or '[none]'}")
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
    print(f"rows total        : {len(rows)}")
    print(f"already completed : {len(rows) - len(pending)}")
    print(f"pending           : {len(pending)}")
    print("prompt protocol   : FIXED training-time compressor prompt")
    print("input truncation  : NEVER")
    print("=" * 72)

    if not pending:
        print("Nothing to do.")
        return

    model, tokenizer = load_model(args)

    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as fout:
        for row in tqdm(pending, desc="Compressing"):
            sid = str(row["id"])
            sseed = sample_seed(args.seed, sid)
            seed_everything(sseed)

            raw_output, prompt_tokens, generated_tokens = generate_one(
                model=model,
                tokenizer=tokenizer,
                context=row["context"],
                device=args.device,
                thinking=args.thinking,
                generation_kwargs=gen_cfg,
                context_limit=context_limit,
            )

            parsed, parse_mode = extract_json_object(raw_output)
            valid_format = validate_structure(parsed)

            source_tokens = int(row["source_tokens"])

            if parsed is not None and isinstance(parsed.get("summary"), str):
                backbone_tokens = count_tokens(tokenizer, parsed["summary"])
                compression_ratio = (
                    backbone_tokens / source_tokens if source_tokens > 0 else None
                )
            else:
                backbone_tokens = None
                compression_ratio = None

            full_tokens = generated_tokens
            full_compression_ratio = (
                full_tokens / source_tokens if source_tokens > 0 else None
            )

            result = {
                "id": sid,
                "output": parsed,
                "raw_output": raw_output,
                "valid_format": valid_format,
                "parse_mode": parse_mode,
                "source_tokens": source_tokens,
                "prompt_tokens": prompt_tokens,
                "backbone_tokens": backbone_tokens,
                "full_tokens": full_tokens,
                "compression_ratio": compression_ratio,
                "full_compression_ratio": full_compression_ratio,
                "sample_seed": sseed,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()