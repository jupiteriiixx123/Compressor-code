#!/usr/bin/env python3
"""
Model-agnostic deterministic binary LLM judge.

- Replace the backbone by changing the model path.
- Never calls generate().
- Scores fixed labels "0" (incorrect) and "1" (correct).
- Uses one batched next-token forward pass when both labels are single tokens.
- Falls back to complete candidate-string conditional log-probabilities when
  either label is multi-token.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class BinaryJudgeResult:
    label: int
    p0: float
    p1: float
    logp0: float
    logp1: float
    candidate_mass: Optional[float]
    scoring_mode: str


def _torch_dtype(name: str):
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _normalize_device_map(value: str):
    value = str(value)
    if value == "auto":
        return "auto"
    if value in {"cpu", "mps"} or value.startswith("cuda"):
        return {"": value}
    raise ValueError(
        "--judge-device-map must be 'auto', 'cpu', 'mps', or e.g. 'cuda:0'."
    )


class BinaryJudge:
    FALSE_LABEL = "0"
    TRUE_LABEL = "1"

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bf16",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.dtype_name = dtype
        self.device_map_arg = device_map
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        load_kwargs = dict(
            config=config,
            torch_dtype=_torch_dtype(dtype),
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            device_map=_normalize_device_map(device_map),
        )
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )
        self.model.eval()

        self.false_ids = self.tokenizer.encode(
            self.FALSE_LABEL,
            add_special_tokens=False,
        )
        self.true_ids = self.tokenizer.encode(
            self.TRUE_LABEL,
            add_special_tokens=False,
        )
        if not self.false_ids or not self.true_ids:
            raise RuntimeError("Judge labels 0/1 produced empty token sequences.")

        self.single_token_fast_path = (
            len(self.false_ids) == 1 and len(self.true_ids) == 1
        )

    def render_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _input_device(self) -> torch.device:
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        raise RuntimeError("Could not determine judge input device.")

    @torch.inference_mode()
    def score(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> BinaryJudgeResult:
        return self.score_batch(
            [{"system_prompt": system_prompt, "user_prompt": user_prompt}]
        )[0]

    @torch.inference_mode()
    def score_batch(
        self,
        items: Sequence[Dict[str, str]],
    ) -> List[BinaryJudgeResult]:
        if not items:
            return []

        rendered = [
            self.render_prompt(x["system_prompt"], x["user_prompt"])
            for x in items
        ]

        if self.single_token_fast_path:
            return self._score_batch_single_token(rendered)

        return [self._score_multitoken(text) for text in rendered]

    @torch.inference_mode()
    def _score_batch_single_token(
        self,
        rendered_prompts: Sequence[str],
    ) -> List[BinaryJudgeResult]:
        batch = self.tokenizer(
            list(rendered_prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        device = self._input_device()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = self.model(**batch, use_cache=False)
        next_logits = outputs.logits[:, -1, :].float()

        id0 = self.false_ids[0]
        id1 = self.true_ids[0]

        pair_logits = next_logits[:, [id0, id1]]
        pair_log_norm = torch.logsumexp(pair_logits, dim=-1, keepdim=True)
        pair_probs = torch.exp(pair_logits - pair_log_norm)

        vocab_log_norm = torch.logsumexp(next_logits, dim=-1)
        logp0 = next_logits[:, id0] - vocab_log_norm
        logp1 = next_logits[:, id1] - vocab_log_norm
        candidate_mass = torch.exp(
            torch.logsumexp(pair_logits, dim=-1) - vocab_log_norm
        )

        labels = (pair_logits[:, 1] > pair_logits[:, 0]).long()

        results = []
        for i in range(len(rendered_prompts)):
            results.append(
                BinaryJudgeResult(
                    label=int(labels[i].item()),
                    p0=float(pair_probs[i, 0].item()),
                    p1=float(pair_probs[i, 1].item()),
                    logp0=float(logp0[i].item()),
                    logp1=float(logp1[i].item()),
                    candidate_mass=float(candidate_mass[i].item()),
                    scoring_mode="single_token_next_logit",
                )
            )
        return results

    @torch.inference_mode()
    def _candidate_sequence_logprob(
        self,
        prompt_ids: List[int],
        candidate_ids: List[int],
    ) -> float:
        device = self._input_device()
        ids = torch.tensor(
            [prompt_ids + candidate_ids],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(ids)

        outputs = self.model(
            input_ids=ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[0].float()

        start = len(prompt_ids)
        if start == 0:
            raise RuntimeError("Rendered judge prompt tokenized to zero tokens.")

        total = torch.tensor(0.0, device=logits.device)
        for j, target_id in enumerate(candidate_ids):
            token_logits = logits[start - 1 + j]
            total = total + torch.log_softmax(token_logits, dim=-1)[target_id]

        return float(total.item())

    def _score_multitoken(self, rendered_prompt: str) -> BinaryJudgeResult:
        prompt_ids = self.tokenizer.encode(
            rendered_prompt,
            add_special_tokens=False,
        )
        logp0 = self._candidate_sequence_logprob(prompt_ids, self.false_ids)
        logp1 = self._candidate_sequence_logprob(prompt_ids, self.true_ids)

        pair = torch.tensor([logp0, logp1], dtype=torch.float64)
        probs = torch.softmax(pair, dim=0)

        return BinaryJudgeResult(
            label=int(logp1 > logp0),
            p0=float(probs[0].item()),
            p1=float(probs[1].item()),
            logp0=logp0,
            logp1=logp1,
            candidate_mass=None,
            scoring_mode="candidate_sequence_logprob",
        )

    def metadata(self) -> Dict[str, Any]:
        cfg = self.model.config
        chat_template = getattr(self.tokenizer, "chat_template", None)
        chat_template_hash = None
        if chat_template:
            chat_template_hash = hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest()

        return {
            "model_path": self.model_path,
            "model_name_or_path": getattr(cfg, "_name_or_path", None),
            "model_type": getattr(cfg, "model_type", None),
            "model_commit_hash": getattr(cfg, "_commit_hash", None),
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "chat_template_sha256": chat_template_hash,
            "dtype": self.dtype_name,
            "device_map": self.device_map_arg,
            "attn_implementation": self.attn_implementation,
            "false_label": self.FALSE_LABEL,
            "true_label": self.TRUE_LABEL,
            "false_label_token_ids": self.false_ids,
            "true_label_token_ids": self.true_ids,
            "single_token_fast_path": self.single_token_fast_path,
            "decision_rule": "argmax P(label | judge prompt), labels={0,1}",
            "generation_used": False,
            "sampling_used": False,
        }
