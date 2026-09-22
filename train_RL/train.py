#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.advantages import GDPOAdvantage, GRPOAdvantage
from utils.io_utils import read_jsonl
from utils.rewards import REWARD_NAMES, RewardEngine


SYSTEM_PROMPT = (
    'You are a long-context compression model. Compress the document into a coherent '
    'summary with expandable semantic information blocks. Output valid JSON with exactly '
    'two keys: "summary" and "link". Use ASCII citation anchors in the form [cite_n]. '
    'The summary length, citation count, block count, block length, and compression ratio '
    'must adapt freely to the document\'s information content.'
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--policy-data", required=True)
    p.add_argument("--reward-data", required=True)
    p.add_argument("--stage1-model", required=True)
    p.add_argument("--reader-model", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--advantage", choices=["grpo", "gdpo"], default="gdpo")
    p.add_argument("--reward-weights", type=float, nargs=4, default=[1.0, 2.0, 6.0, 0.2])

    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--update-epochs", type=int, default=2)

    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--clip-eps", type=float, default=0.2)

    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--target-ratio", type=float, default=0.30)
    p.add_argument("--empty-support-threshold", type=float, default=0.50)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--reader-device", default="cuda:1")
    p.add_argument("--reader-batch-size", type=int, default=16)
    p.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")

    p.add_argument("--monitor-size", type=int, default=3)
    p.add_argument("--monitor-every", type=int, default=20)

    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume-from", default=None)

    p.add_argument("--debug", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def render_prompt(tokenizer, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def trim_rollout(tokenizer, sequence: torch.Tensor, prompt_len: int) -> Dict:
    completion_ids = sequence[prompt_len:]
    eos = (completion_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
    completion_len = int(eos[0].item()) + 1 if len(eos) else int(completion_ids.numel())

    sequence = sequence[: prompt_len + completion_len].detach().cpu()
    completion_ids = sequence[prompt_len:]

    return {
        "input_ids": sequence,
        "prompt_len": prompt_len,
        # Keep exact decoded whitespace so token-to-text cite spans remain aligned.
        "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "completion_tokens": completion_len,
    }


def generate_group(model, tokenizer, context, args, device) -> List[Dict]:
    prompt = render_prompt(tokenizer, context)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=0,
            num_return_sequences=args.group_size,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    model.train()

    return [trim_rollout(tokenizer, seq, prompt_len) for seq in outputs]


def generate_greedy(model, tokenizer, context, max_new_tokens, device) -> Dict:
    prompt = render_prompt(tokenizer, context)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    model.eval()
    with torch.inference_mode():
        sequence = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    model.train()

    return trim_rollout(tokenizer, sequence, prompt_len)


def completion_logps(model, rollout: Dict, device) -> torch.Tensor:
    input_ids = rollout["input_ids"].unsqueeze(0).to(device)
    prompt_len = rollout["prompt_len"]
    completion_len = input_ids.shape[1] - prompt_len

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        logits_to_keep=completion_len + 1,
    )

    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, -completion_len:]
    logps = torch.log_softmax(logits.float(), dim=-1)

    return logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1).squeeze(0)


def attach_old_logps(model, rollout: Dict, device) -> None:
    model.eval()
    with torch.inference_mode():
        rollout["old_logps"] = completion_logps(model, rollout, device).detach().cpu()
    model.train()


def policy_loss(model, rollout: Dict, advantage, clip_eps, device):
    current_logps = completion_logps(model, rollout, device)
    old_logps = rollout["old_logps"].to(device)
    ratio = torch.exp(current_logps - old_logps)

    advantage = torch.as_tensor(
        advantage,
        dtype=current_logps.dtype,
        device=device,
    )

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    loss = -torch.minimum(unclipped, clipped).mean()

    return loss, ratio.detach().float().cpu()


def cite_value_token_ranges(tokenizer, rollout: Dict) -> Dict[str, tuple[int, int]]:
    """Map each JSON link value to completion-token [start, end) indices."""
    completion = rollout["completion"]
    try:
        obj = json.loads(completion.strip())
    except Exception:
        return {}

    if not isinstance(obj, dict) or not isinstance(obj.get("link"), dict):
        return {}

    try:
        encoded = tokenizer(
            completion,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        text_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except Exception:
        return {}

    generated_ids = rollout["input_ids"][rollout["prompt_len"]:].tolist()
    if generated_ids[:len(text_ids)] != list(text_ids):
        return {}

    decoder = json.JSONDecoder()
    ranges: Dict[str, tuple[int, int]] = {}

    for key, expected_value in obj["link"].items():
        if not re.fullmatch(r"cite_\d+", str(key)) or not isinstance(expected_value, str):
            continue

        pattern = re.compile(rf'"{re.escape(str(key))}"\s*:\s*')
        match = pattern.search(completion)
        if match is None:
            continue

        char_start = match.end()
        try:
            parsed_value, char_end = decoder.raw_decode(completion, char_start)
        except Exception:
            continue
        if parsed_value != expected_value:
            continue

        token_indices = [
            i for i, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end
        ]
        if token_indices:
            ranges[str(key)] = (token_indices[0], token_indices[-1] + 1)

    return ranges


def build_token_advantage(
    tokenizer,
    rollout: Dict,
    global_advantage: torch.Tensor,
    cite_credits: Dict[str, float],
    utility_weight: float,
    device,
) -> torch.Tensor:
    """
    Keep the original rollout-level GRPO/GDPO advantage everywhere.
    On cite-detail value tokens only, add signed marginal utility:
        global_adv + utility_weight * (U_full - U_without_cite)
    This is intentionally a local correction, not another normalized reward dimension.
    """
    length = int(rollout["old_logps"].numel())
    token_adv = torch.full(
        (length,),
        float(global_advantage.item()),
        dtype=torch.float32,
        device=device,
    )

    if not cite_credits:
        return token_adv

    ranges = cite_value_token_ranges(tokenizer, rollout)
    for cite_key, delta in cite_credits.items():
        if cite_key not in ranges:
            continue
        start, end = ranges[cite_key]
        end = min(end, length)
        if start < end:
            token_adv[start:end] += float(utility_weight) * float(delta)

    return token_adv


def grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5


def write_jsonl(handle, row: Dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def evaluate_monitor(policy, tokenizer, reward_engine, rows, args, device, step):
    rewards = []
    backbone_ratios = []
    full_ratios = []
    qa_accuracies = []

    for row in rows:
        rollout = generate_greedy(
            policy,
            tokenizer,
            row["context"],
            args.max_new_tokens,
            device,
        )
        rewards.append(
            reward_engine.score(
                sample_id=row["id"],
                completion=rollout["completion"],
                source_tokens=row["source_tokens"],
                compute_cite_credit=False,
            )
        )
        stats = reward_engine.compression_stats(
            rollout["completion"],
            row["source_tokens"],
        )
        backbone_ratios.append(stats["compression_ratio"])
        full_ratios.append(stats["full_compression_ratio"])
        qa_accuracies.append(
            reward_engine.qa_accuracy(row["id"], rollout["completion"])
        )

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    mean_rewards = reward_tensor.mean(dim=0)
    weights = torch.tensor(args.reward_weights, dtype=torch.float32)

    metrics = {
        "step": step,
        "total_reward": (reward_tensor * weights).sum(dim=-1).mean().item(),
        # Compression ratio now means backbone-summary tokens only.
        "compression_ratio": sum(backbone_ratios) / len(backbone_ratios),
        "full_compression_ratio": sum(full_ratios) / len(full_ratios),
        "qa_accuracy": sum(qa_accuracies) / len(qa_accuracies),
    }
    for i, name in enumerate(REWARD_NAMES):
        metrics[name] = mean_rewards[i].item()
    return metrics


def save_checkpoint(policy, optimizer, output_dir: Path, step, next_epoch, next_batch):
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    policy.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": next_epoch,
            "batch": next_batch,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    return checkpoint_dir


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy_device = torch.device(args.policy_device)
    policy_rows = list(read_jsonl(args.policy_data))

    tokenizer = AutoTokenizer.from_pretrained(args.stage1_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.stage1_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(policy_device)
    base_model.config.use_cache = False

    if args.resume_from:
        policy = PeftModel.from_pretrained(
            base_model,
            args.resume_from,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        policy = get_peft_model(base_model, lora_config)

    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    policy.enable_input_require_grads()
    policy.train()

    reward_engine = RewardEngine(
        reward_data=args.reward_data,
        reader_model=args.reader_model,
        policy_tokenizer=tokenizer,
        reader_device=args.reader_device,
        reader_batch_size=args.reader_batch_size,
        target_ratio=args.target_ratio,
        empty_support_threshold=args.empty_support_threshold,
        attn_implementation=args.attn_implementation,
    )

    advantage_fn = (
        GRPOAdvantage(args.reward_weights)
        if args.advantage == "grpo"
        else GDPOAdvantage(args.reward_weights)
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    start_epoch = 0
    start_batch = 0

    if args.resume_from:
        state = torch.load(
            Path(args.resume_from) / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        start_epoch = state["epoch"]
        start_batch = state["batch"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resumed from {args.resume_from} at step={step}")

    monitor_rng = random.Random(args.seed + 1)
    monitor_rows = monitor_rng.sample(
        policy_rows,
        min(args.monitor_size, len(policy_rows)),
    ) if args.monitor_size > 0 else []

    mode = "a" if args.resume_from else "w"
    train_log = (output_dir / "train_metrics.jsonl").open(mode, encoding="utf-8")
    monitor_log = (output_dir / "monitor_metrics.jsonl").open(mode, encoding="utf-8")
    debug_log = (output_dir / "reward_debug.jsonl").open(mode, encoding="utf-8") if args.debug else None

    config = vars(args).copy()
    config["monitor_ids"] = [row["id"] for row in monitor_rows]
    config["compression_ratio_definition"] = "summary/backbone tokens divided by source tokens; link/cite detail tokens excluded"
    config["utility_reward_definition"] = "EMPTY-calibrated soft binary evidence sufficiency for the known gold answer"
    config["cite_credit_definition"] = "leave-one-cite-detail-out marginal utility; summary and citation anchor remain"
    (output_dir / "stage2_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_batches = math.ceil(len(policy_rows) / args.batch_size)
    total_steps = args.epochs * total_batches
    progress_total = min(total_steps, args.max_steps) if args.max_steps is not None else total_steps
    progress = tqdm(
        total=progress_total,
        initial=step,
        desc="Stage-2",
        dynamic_ncols=True,
    )

    if step == 0 and monitor_rows:
        metrics = evaluate_monitor(
            policy, tokenizer, reward_engine,
            monitor_rows, args, policy_device, step=0,
        )
        write_jsonl(monitor_log, metrics)
        tqdm.write(
            f"monitor step=0 semantic={metrics['semantic']:.4f} "
            f"utility_support={metrics['utility']:.4f} "
            f"qa_accuracy={metrics['qa_accuracy']:.4f} "
            f"backbone_ratio={metrics['compression_ratio']:.4f}"
        )

    for epoch in range(start_epoch, args.epochs):
        epoch_rows = list(policy_rows)
        random.Random(args.seed + epoch).shuffle(epoch_rows)

        first_batch = start_batch if epoch == start_epoch else 0

        for batch_idx in range(first_batch, total_batches):
            step_start = time.perf_counter()
            start = batch_idx * args.batch_size
            batch = epoch_rows[start:start + args.batch_size]

            batch_rollouts = []
            reward_matrix = []
            batch_cite_credits = []
            batch_compression_stats = []

            # 1. Sample rollout group and freeze behavior-policy log-probs.
            for row in batch:
                rollouts = generate_group(policy, tokenizer, row["context"], args, policy_device)
                for rollout in rollouts:
                    attach_old_logps(policy, rollout, policy_device)

                rewards = []
                cite_credit_rows = []
                compression_rows = []

                for rollout_idx, rollout in enumerate(rollouts):
                    reward, aux = reward_engine.score(
                        sample_id=row["id"],
                        completion=rollout["completion"],
                        source_tokens=row["source_tokens"],
                        compute_cite_credit=True,
                        return_details=args.debug,
                        return_aux=True,
                    )
                    rewards.append(reward)
                    cite_credit_rows.append(aux["cite_credits"])
                    compression_rows.append(aux["compression"])

                    if args.debug:
                        write_jsonl(debug_log, {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "sample_id": row["id"],
                            "rollout": rollout_idx + 1,
                            "completion": rollout["completion"],
                            "completion_tokens": rollout["completion_tokens"],
                            "source_tokens": row["source_tokens"],
                            "backbone_tokens": aux["compression"]["backbone_tokens"],
                            "compression_ratio": aux["compression"]["compression_ratio"],
                            "full_compression_ratio": aux["compression"]["full_compression_ratio"],
                            "rewards": {name: reward[i] for i, name in enumerate(REWARD_NAMES)},
                            **aux,
                        })

                batch_rollouts.append(rollouts)
                reward_matrix.append(rewards)
                batch_cite_credits.append(cite_credit_rows)
                batch_compression_stats.append(compression_rows)

            rewards = torch.tensor(reward_matrix, dtype=torch.float32, device=policy_device)
            advantages = advantage_fn(rewards).detach()

            # Fine-grained credit: retain the original global advantage, then locally
            # adjust only cite-detail value tokens by signed marginal utility.
            token_advantages = []
            for b, rollouts in enumerate(batch_rollouts):
                row_advantages = []
                for g, rollout in enumerate(rollouts):
                    row_advantages.append(
                        build_token_advantage(
                            tokenizer=tokenizer,
                            rollout=rollout,
                            global_advantage=advantages[b, g],
                            cite_credits=batch_cite_credits[b][g],
                            utility_weight=args.reward_weights[1],
                            device=policy_device,
                        )
                    )
                token_advantages.append(row_advantages)

            # 2. Multi-update clipped GRPO/GDPO.
            n_rollouts = len(batch) * args.group_size
            last_loss = 0.0
            last_grad = 0.0
            last_ratio = None

            for _ in range(args.update_epochs):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                ratios = []

                for b, rollouts in enumerate(batch_rollouts):
                    for g, rollout in enumerate(rollouts):
                        loss, ratio = policy_loss(
                            policy,
                            rollout,
                            token_advantages[b][g],
                            args.clip_eps,
                            policy_device,
                        )
                        (loss / n_rollouts).backward()
                        losses.append(float(loss.detach()))
                        ratios.append(ratio)

                last_grad = grad_norm(policy)
                optimizer.step()

                last_loss = sum(losses) / len(losses)
                last_ratio = torch.cat(ratios)

            step += 1

            # 3. Minimal training log for later plots.
            mean_rewards = rewards.mean(dim=(0, 1)).detach().cpu()
            weights = torch.tensor(args.reward_weights, device=policy_device)
            total_reward = (rewards * weights).sum(dim=-1).mean().item()
            clip_fraction = (
                (last_ratio < 1 - args.clip_eps) | (last_ratio > 1 + args.clip_eps)
            ).float().mean().item()

            backbone_ratios = [
                stats["compression_ratio"]
                for rows in batch_compression_stats
                for stats in rows
            ]
            full_ratios = [
                stats["full_compression_ratio"]
                for rows in batch_compression_stats
                for stats in rows
            ]

            metrics = {
                "epoch": epoch + 1,
                "step": step,
                "loss": last_loss,
                "grad_norm": last_grad,
                "total_reward": total_reward,
                "ratio_mean": last_ratio.mean().item(),
                "clip_fraction": clip_fraction,
                "compression_ratio": sum(backbone_ratios) / len(backbone_ratios),
                "full_compression_ratio": sum(full_ratios) / len(full_ratios),
                "step_seconds": time.perf_counter() - step_start,
            }
            for i, name in enumerate(REWARD_NAMES):
                metrics[name] = mean_rewards[i].item()
            write_jsonl(train_log, metrics)

            progress.set_postfix(
                sem=f"{metrics['semantic']:.3f}",
                util=f"{metrics['utility']:.3f}",
                ratio=f"{metrics['ratio_mean']:.3f}",
                grad=f"{metrics['grad_norm']:.2e}",
            )
            progress.update(1)

            # 4. Fixed monitor set: same articles, greedy generation.
            if monitor_rows and step % args.monitor_every == 0:
                monitor_metrics = evaluate_monitor(
                    policy, tokenizer, reward_engine,
                    monitor_rows, args, policy_device, step,
                )
                write_jsonl(monitor_log, monitor_metrics)
                tqdm.write(
                    f"monitor step={step} "
                    f"semantic={monitor_metrics['semantic']:.4f} "
                    f"utility_support={monitor_metrics['utility']:.4f} "
                    f"qa_accuracy={monitor_metrics['qa_accuracy']:.4f} "
                    f"backbone_ratio={monitor_metrics['compression_ratio']:.4f}"
                )

            # 5. Lightweight LoRA + optimizer checkpoint.
            if args.save_every > 0 and step % args.save_every == 0:
                if batch_idx + 1 < total_batches:
                    next_epoch, next_batch = epoch, batch_idx + 1
                else:
                    next_epoch, next_batch = epoch + 1, 0

                checkpoint_dir = save_checkpoint(
                    policy,
                    optimizer,
                    output_dir,
                    step,
                    next_epoch,
                    next_batch,
                )
                tqdm.write(f"saved {checkpoint_dir}")

            if args.max_steps is not None and step >= args.max_steps:
                break

        start_batch = 0
        if args.max_steps is not None and step >= args.max_steps:
            break

    progress.close()
    train_log.close()
    monitor_log.close()
    if debug_log is not None:
        debug_log.close()

    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nStage-2 LoRA saved to: {output_dir}")


if __name__ == "__main__":
    main()
