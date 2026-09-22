#!/usr/bin/env python3
"""
train_sft_lora_3gpu.py

Qwen3-4B LoRA SFT for long-context compression.

Launch examples:
  Single GPU:
    python train_sft_lora_3gpu.py ...

  3-GPU DDP:
    CUDA_VISIBLE_DEVICES=0,1,2 \
    torchrun --standalone --nproc_per_node=3 train_sft_lora_3gpu.py ...

Core policy:
- Qwen thinking disabled.
- TRL prompt-completion SFT.
- completion_only_loss=True.
- No silent truncation.
- Samples longer than --max-length are excluded and recorded.

完整启动命令：
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun \
  --standalone \
  --nproc_per_node=3 \
  scripts/train_sft_lora.py \
  --data training_data/sft_train.jsonl \
  --model /public/home/hh_hh/LoRA/models/Qwen3-4B \
  --output-dir outputs/sft_lora_600_v1 \
  --max-length 40960 \
  --num-train-epochs 3 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 3 \
  --attn-implementation flash_attention_2 \
  --save-steps 50 \
  --logging-steps 1

启动命令:
CUDA_VISIBLE_DEVICES=0,1,4 \
torchrun \
  --standalone \
  --nproc_per_node=3 \
  scripts/train_sft_lora.py \
  --data training_data/sft_train.jsonl \
  --model /public/home/hh_hh/LoRA/models/Qwen3-4B \
  --output-dir outputs/sft_lora_600_v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    # Formal LoRA defaults.
    p.add_argument("--max-length", type=int, default=40960)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=3)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # LoRA.
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Runtime.
    p.add_argument(
        "--attn-implementation",
        choices=["flash_attention_2", "sdpa"],
        default="sdpa",
    )
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", default=None)

    return p.parse_args()


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read local JSONL directly; avoids HF Datasets cache locks on NFS."""
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e

    if not rows:
        raise RuntimeError(f"No examples found in {path}")

    return rows


def validate_example(row: Dict[str, Any]) -> None:
    doc_id = row.get("id", "<missing-id>")
    messages = row.get("messages")

    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{doc_id}: expected exactly 3 messages")

    roles = [m.get("role") for m in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"{doc_id}: invalid roles: {roles}")

    for message in messages:
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{doc_id}: invalid message content")

    try:
        target = json.loads(messages[-1]["content"])
    except json.JSONDecodeError as e:
        raise ValueError(f"{doc_id}: assistant target is not valid JSON") from e

    if not isinstance(target, dict) or set(target) != {"summary", "link"}:
        raise ValueError(
            f"{doc_id}: assistant target must contain exactly summary and link"
        )


def render_example(row: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Convert messages -> prompt/completion.

    The prompt uses Qwen's own chat template with enable_thinking=False.
    TRL later handles tokenization and completion-only loss masking.
    """
    messages = row["messages"]
    completion = messages[-1]["content"]

    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    eos = tokenizer.eos_token or ""
    training_text = prompt + completion
    if eos and not training_text.endswith(eos):
        training_text += eos

    prompt_ids = tokenizer(prompt).input_ids
    total_ids = tokenizer(training_text).input_ids

    if total_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"{row['id']}: prompt is not a token prefix of prompt+completion"
        )

    return {
        "id": row["id"],
        "prompt": prompt,
        "completion": completion,
        "total_tokens": len(total_ids),
    }


def build_dataset(
    rows: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """
    Keep every sample that fits max_length.
    Overlong samples are excluded; nothing is truncated.
    """
    kept = []
    excluded = []
    seen = set()

    for row in rows:
        validate_example(row)

        doc_id = row["id"]
        if doc_id in seen:
            raise ValueError(f"Duplicate id: {doc_id}")
        seen.add(doc_id)

        item = render_example(row, tokenizer)

        if item["total_tokens"] > max_length:
            excluded.append(
                {"id": item["id"], "total_tokens": item["total_tokens"]}
            )
        else:
            kept.append(item)

    if not kept:
        raise RuntimeError("No training examples remain after length filtering")

    return Dataset.from_list(kept), excluded


def check_model_context(model_path: str, max_length: int) -> None:
    """Do not exceed native context unless RoPE scaling is already configured."""
    config = AutoConfig.from_pretrained(model_path)

    native_limit = getattr(config, "max_position_embeddings", None)
    rope_scaling = getattr(config, "rope_scaling", None)

    if native_limit is not None and max_length > native_limit and not rope_scaling:
        raise ValueError(
            f"--max-length={max_length} exceeds "
            f"max_position_embeddings={native_limit}, "
            "and rope_scaling is not configured"
        )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    check_model_context(args.model, args.max_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    dataset, excluded = build_dataset(
        load_jsonl(args.data),
        tokenizer,
        args.max_length,
    )

    if is_rank0():
        lengths = dataset["total_tokens"]

        print("\nDataset")
        print("-------")
        print("Kept examples:    ", len(dataset))
        print("Excluded examples:", len(excluded))
        print("Min tokens:       ", min(lengths))
        print("Max tokens:       ", max(lengths))

        if excluded:
            print("\nExcluded overlong samples:")
            for item in excluded:
                print(f"  {item['id']}: {item['total_tokens']}")

        (args.output_dir / "selected_train_ids.json").write_text(
            json.dumps(list(dataset["id"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (args.output_dir / "excluded_overlong.json").write_text(
            json.dumps(excluded, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # TRL only needs these two columns.
    train_dataset = dataset.select_columns(["prompt", "completion"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_args = SFTConfig(
        output_dir=str(args.output_dir),

        # Objective.
        completion_only_loss=True,
        max_length=args.max_length,
        packing=False,

        # Optimization.
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        # Long-context memory.
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Multi-GPU DDP.
        group_by_length=True,
        ddp_find_unused_parameters=False,

        # Logging / checkpoints.
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,

        use_liger_kernel=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    if trainer.is_world_process_zero():
        world_size = trainer.args.world_size
        effective_batch = (
            world_size
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )

        trainable = sum(
            p.numel()
            for p in trainer.model.parameters()
            if p.requires_grad
        )
        total = sum(p.numel() for p in trainer.model.parameters())

        print("\nLoRA SFT")
        print("--------")
        print("World size:       ", world_size)
        print("Examples:         ", len(train_dataset))
        print("Max length:       ", args.max_length)
        print("Batch / GPU:      ", args.per_device_train_batch_size)
        print("Grad accumulation:", args.gradient_accumulation_steps)
        print("Effective batch:  ", effective_batch)
        print("Learning rate:    ", args.learning_rate)
        print("Epochs:           ", args.num_train_epochs)
        print("Attention:        ", args.attn_implementation)
        print(
            f"Trainable params:  {trainable:,}/{total:,} "
            f"({100 * trainable / total:.4f}%)"
        )
        print()

    result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # Trainer handles distributed model saving internally.
    trainer.save_model(str(args.output_dir))

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output_dir))

        metrics = dict(result.metrics)
        metrics.update(
            {
                "train_examples": len(train_dataset),
                "excluded_overlong": len(excluded),
                "world_size": trainer.args.world_size,
                "effective_batch_size": (
                    trainer.args.world_size
                    * args.per_device_train_batch_size
                    * args.gradient_accumulation_steps
                ),
            }
        )

        (args.output_dir / "run_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nLoRA training finished:", args.output_dir)


if __name__ == "__main__":
    main()

