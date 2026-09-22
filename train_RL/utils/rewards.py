from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io_utils import read_jsonl


REWARD_NAMES = ("semantic", "utility", "format", "budget")


class RewardEngine:
    """
    Reward vector order:
        [semantic, utility, format, budget]

    Semantic:
        binary next-token coverage judgment over atomic information units.

    Utility (training):
        binary next-token evidence-sufficiency judgment for the known gold answer.
        The score is soft P(1), calibrated against an EMPTY-compression baseline.

    Cite credit:
        leave-one-cite-out marginal utility. Only the expandable cite detail block is
        removed; the summary/backbone sentence and [cite_n] anchor remain untouched.
    """

    CANDIDATE_MASS_THRESHOLD = 0.90

    def __init__(
        self,
        reward_data: str,
        reader_model: str,
        policy_tokenizer,
        reader_device: str = "cuda:1",
        reader_batch_size: int = 16,
        target_ratio: float = 0.30,
        empty_support_threshold: float = 0.50,
        attn_implementation: str = "sdpa",
    ):
        self.data = {row["id"]: row for row in read_jsonl(reward_data)}
        self.policy_tokenizer = policy_tokenizer
        self.target_ratio = target_ratio
        self.empty_support_threshold = empty_support_threshold
        self.reader_device = torch.device(reader_device)
        self.reader_batch_size = reader_batch_size
        self._empty_support_cache: Dict[str, List[Dict[str, Any]]] = {}

        self.reader_tokenizer = AutoTokenizer.from_pretrained(reader_model, use_fast=True)
        if self.reader_tokenizer.pad_token_id is None:
            self.reader_tokenizer.pad_token = self.reader_tokenizer.eos_token
        self.reader_tokenizer.padding_side = "left"

        self.reader = AutoModelForCausalLM.from_pretrained(
            reader_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).to(self.reader_device)
        self.reader.eval()

        self.binary_token_ids = self._single_token_ids(["0", "1"])
        self.mcq_token_ids = self._single_token_ids(["A", "B", "C", "D"])

    def score(
        self,
        sample_id: str,
        completion: str,
        source_tokens: int,
        compute_cite_credit: bool = False,
        return_details: bool = False,
        return_aux: bool = False,
    ):
        meta = self.data[sample_id]
        reader_completion = self.reader_view(completion)

        if return_details:
            semantic, semantic_details = self.semantic_reward(
                reader_completion,
                meta["information_units"],
                return_details=True,
            )
        else:
            semantic = self.semantic_reward(
                reader_completion,
                meta["information_units"],
            )

        utility, utility_aux = self.utility_reward(
            sample_id=sample_id,
            completion=completion,
            probes=meta["probes"],
            compute_cite_credit=compute_cite_credit,
            return_details=return_details,
        )

        format_score = self.format_reward(completion)
        compression = self.compression_stats(completion, source_tokens)
        budget = self.budget_reward_from_ratio(compression["compression_ratio"])
        rewards = [semantic, utility, format_score, budget]

        if not (return_details or return_aux):
            return rewards

        aux = {
            "cite_credits": utility_aux["cite_credits"],
            "utility_meta": utility_aux["meta"],
            "compression": compression,
        }

        if return_details:
            aux.update({
                "semantic": semantic_details,
                "utility": utility_aux["details"],
                "utility_ablation": utility_aux["utility_ablation"],
                "cite_ablation": utility_aux["cite_ablation"],
                "reader_candidates": {
                    "binary": self._candidate_meta(["0", "1"], self.binary_token_ids),
                    "mcq": self._candidate_meta(["A", "B", "C", "D"], self.mcq_token_ids),
                },
            })

        return rewards, aux

    def semantic_reward(
        self,
        reader_completion: str,
        information_units: List[Dict[str, Any]],
        return_details: bool = False,
    ):
        prompts = [
            self._render_judge_prompt(
                system=(
                    "You are a strict information-coverage evaluator. "
                    "Judge only from the supplied compression."
                ),
                user=f"""COMPRESSION:
<<<COMPRESSION
{reader_completion}
COMPRESSION>>>

CLAIM:
{unit['claim']}

Decide whether the factual content of the claim can be fully recovered from the compression.

0 = not recoverable, incomplete, uncertain, or contradicted
1 = fully recoverable

Answer with exactly one digit: 0 or 1.""",
            )
            for unit in information_units
        ]

        if return_details:
            labels, scores = self._classify_next_token(
                prompts,
                self.binary_token_ids,
                return_scores=True,
            )
            details = []
            for unit, label, score in zip(information_units, labels, scores):
                details.append({
                    "claim": unit["claim"],
                    "evidence": unit.get("evidence", ""),
                    "prediction": int(label),
                    "candidate_probs": {
                        "0": score["candidate_probs"][0],
                        "1": score["candidate_probs"][1],
                    },
                    "candidate_mass": score["candidate_mass"],
                })
            return sum(labels) / len(labels), details

        labels = self._classify_next_token(
            prompts,
            self.binary_token_ids,
        )
        return sum(labels) / len(labels)

    def utility_reward(
        self,
        sample_id: str,
        completion: str,
        probes: List[Dict[str, Any]],
        compute_cite_credit: bool = False,
        return_details: bool = False,
    ):
        empty_scores = self._get_empty_support_scores(sample_id, probes)

        full_view = self.reader_view(completion)
        full_scores = self._support_scores(full_view, probes)

        valid_mask = [
            self._empty_probe_is_valid(score)
            for score in empty_scores
        ]
        full_calibrated = [
            self._calibrated_support(full, empty) if valid else 0.0
            for full, empty, valid in zip(full_scores, empty_scores, valid_mask)
        ]
        utility = self._mean_selected(full_calibrated, valid_mask)

        cite_credits: Dict[str, float] = {}
        cite_ablation: Dict[str, Dict[str, float]] = {}
        utility_ablation: Dict[str, float] = {}

        # Final debug-only expansion: compare the summary/backbone by itself against
        # the eagerly flattened full compression. Keep only aggregate values so the
        # debug log does not grow another per-probe section. Formal training pays no
        # extra Reader cost because return_details=False there.
        if return_details:
            backbone_view = self.reader_view(completion, backbone_only=True)
            backbone_scores = self._support_scores(backbone_view, probes)
            backbone_calibrated = [
                self._calibrated_support(backbone, empty) if valid else 0.0
                for backbone, empty, valid in zip(backbone_scores, empty_scores, valid_mask)
            ]
            backbone_utility = self._mean_selected(backbone_calibrated, valid_mask)
            utility_ablation = {
                "backbone_only": float(backbone_utility),
                "full": float(utility),
                "cite_layer_credit": float(utility - backbone_utility),
            }

        if compute_cite_credit:
            cite_keys = self.cite_keys(completion)
            if cite_keys:
                all_prompts: List[str] = []
                for cite_key in cite_keys:
                    ablated_view = self.reader_view(completion, omit_cite=cite_key)
                    all_prompts.extend(self._support_prompts(ablated_view, probes))

                _, all_scores = self._classify_next_token(
                    all_prompts,
                    self.binary_token_ids,
                    return_scores=True,
                )

                q = len(probes)
                for i, cite_key in enumerate(cite_keys):
                    scores = all_scores[i * q:(i + 1) * q]
                    calibrated = []
                    for j, (score, empty, valid) in enumerate(zip(scores, empty_scores, valid_mask)):
                        if not valid:
                            calibrated.append(0.0)
                        elif score["candidate_mass"] < self.CANDIDATE_MASS_THRESHOLD:
                            # Do not create local cite credit from an unreliable ablation judgment.
                            calibrated.append(full_calibrated[j])
                        else:
                            calibrated.append(self._calibrated_support(score, empty))
                    ablated_utility = self._mean_selected(calibrated, valid_mask)
                    delta = utility - ablated_utility
                    cite_credits[cite_key] = float(delta)
                    if return_details:
                        cite_ablation[cite_key] = {
                            "utility_full": float(utility),
                            "utility_without_cite": float(ablated_utility),
                            "marginal_utility": float(delta),
                        }

        details: List[Dict[str, Any]] = []
        if return_details:
            for probe, empty, full, calibrated, valid in zip(
                probes,
                empty_scores,
                full_scores,
                full_calibrated,
                valid_mask,
            ):
                gold_letter = probe["answer_letter"].upper()
                gold_index = ord(gold_letter) - ord("A")
                details.append({
                    "question": probe["question"],
                    "choices": probe["choices"],
                    "gold": gold_letter,
                    "gold_text": probe["choices"][gold_index],
                    "used_for_reward": bool(valid),
                    "empty_support_prob": empty["candidate_probs"][1],
                    "full_support_prob": full["candidate_probs"][1],
                    "calibrated_support": float(calibrated),
                    "empty_candidate_mass": empty["candidate_mass"],
                    "full_candidate_mass": full["candidate_mass"],
                    "prediction": int(full["candidate_probs"][1] >= full["candidate_probs"][0]),
                    "candidate_probs": {
                        "0": full["candidate_probs"][0],
                        "1": full["candidate_probs"][1],
                    },
                    **({"empty_top_tokens": empty["top_tokens"]} if "top_tokens" in empty else {}),
                    **({"full_top_tokens": full["top_tokens"]} if "top_tokens" in full else {}),
                })

        meta = {
            "total_probes": len(probes),
            "valid_probes": int(sum(valid_mask)),
            "filtered_probes": int(len(probes) - sum(valid_mask)),
            "empty_support_threshold": self.empty_support_threshold,
            "candidate_mass_threshold": self.CANDIDATE_MASS_THRESHOLD,
            "utility_definition": "mean EMPTY-calibrated binary support probability",
        }

        return utility, {
            "cite_credits": cite_credits,
            "cite_ablation": cite_ablation,
            "utility_ablation": utility_ablation,
            "details": details,
            "meta": meta,
        }

    def qa_accuracy(self, sample_id: str, completion: str) -> float:
        """Hard A/B/C/D accuracy for fixed-monitor diagnostics only, not training reward."""
        probes = self.data[sample_id]["probes"]
        reader_completion = self.reader_view(completion)
        prompts = []

        for probe in probes:
            choices = "\n".join(
                f"{chr(ord('A') + i)}. {choice}"
                for i, choice in enumerate(probe["choices"])
            )
            prompts.append(
                self._render_judge_prompt(
                    system=(
                        "You are a frozen reader evaluating a compressed document. "
                        "Use only the supplied compression and answer with one option letter."
                    ),
                    user=f"""COMPRESSION:
<<<COMPRESSION
{reader_completion}
COMPRESSION>>>

QUESTION:
{probe['question']}

CHOICES:
{choices}

Answer with exactly one letter: A, B, C, or D.""",
                )
            )

        predictions = self._classify_next_token(prompts, self.mcq_token_ids)
        correct = 0
        for pred, probe in zip(predictions, probes):
            gold = ord(probe["answer_letter"].upper()) - ord("A")
            correct += int(pred == gold)
        return correct / len(probes)

    def _get_empty_support_scores(
        self,
        sample_id: str,
        probes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if sample_id not in self._empty_support_cache:
            self._empty_support_cache[sample_id] = self._support_scores("[EMPTY]", probes)
        return self._empty_support_cache[sample_id]

    def _support_scores(
        self,
        reader_completion: str,
        probes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompts = self._support_prompts(reader_completion, probes)
        _, scores = self._classify_next_token(
            prompts,
            self.binary_token_ids,
            return_scores=True,
        )
        return scores

    def _support_prompts(
        self,
        reader_completion: str,
        probes: List[Dict[str, Any]],
    ) -> List[str]:
        prompts = []
        for probe in probes:
            choices = "\n".join(
                f"{chr(ord('A') + i)}. {choice}"
                for i, choice in enumerate(probe["choices"])
            )
            gold_letter = probe["answer_letter"].upper()
            gold_index = ord(gold_letter) - ord("A")
            gold_text = probe["choices"][gold_index]

            prompts.append(
                self._render_judge_prompt(
                    system=(
                        "You are a strict evidence-sufficiency evaluator. The correct answer is given to you. "
                        "Do not solve the question from outside knowledge, memorized facts, plausibility, or test-taking priors. "
                        "Judge only whether the supplied compression contains enough evidence to establish the given correct "
                        "answer over the alternatives."
                    ),
                    user=f"""COMPRESSION:
<<<COMPRESSION
{reader_completion}
COMPRESSION>>>

QUESTION:
{probe['question']}

CHOICES:
{choices}

GIVEN CORRECT ANSWER:
{gold_letter}. {gold_text}

Does the compression itself contain enough information to establish that the given correct answer is correct and distinguish it from the alternatives?

0 = insufficient evidence in the compression
1 = sufficient evidence in the compression

Your first and only output token must be exactly 0 or 1.""",
                )
            )
        return prompts

    def _empty_probe_is_valid(self, empty_score: Dict[str, Any]) -> bool:
        return (
            empty_score["candidate_mass"] >= self.CANDIDATE_MASS_THRESHOLD
            and empty_score["candidate_probs"][1] < self.empty_support_threshold
        )

    def _calibrated_support(
        self,
        full_score: Dict[str, Any],
        empty_score: Dict[str, Any],
    ) -> float:
        if full_score["candidate_mass"] < self.CANDIDATE_MASS_THRESHOLD:
            return 0.0

        p_full = float(full_score["candidate_probs"][1])
        p_empty = float(empty_score["candidate_probs"][1])

        # EMPTY maps to 0; a perfect supported answer maps to 1.
        calibrated = (p_full - p_empty) / max(1.0 - p_empty, 1e-8)
        return max(0.0, min(1.0, calibrated))

    @staticmethod
    def _mean_selected(values: List[float], mask: List[bool]) -> float:
        selected = [v for v, keep in zip(values, mask) if keep]
        return sum(selected) / len(selected) if selected else 0.0

    @staticmethod
    def format_reward(completion: str) -> float:
        try:
            obj = json.loads(completion.strip())
        except Exception:
            return 0.0

        if not isinstance(obj, dict) or set(obj) != {"summary", "link"}:
            return 0.0
        if not isinstance(obj["summary"], str) or not obj["summary"].strip():
            return 0.0
        if not isinstance(obj["link"], dict):
            return 0.0

        link = obj["link"]
        for key, value in link.items():
            if not re.fullmatch(r"cite_\d+", str(key)):
                return 0.0
            if not isinstance(value, str) or not value.strip():
                return 0.0

        cited = set(re.findall(r"\[(cite_\d+)\]", obj["summary"]))
        if cited != set(link.keys()):
            return 0.0

        return 1.0

    def compression_stats(self, completion: str, source_tokens: int) -> Dict[str, Any]:
        full_tokens = len(
            self.policy_tokenizer(
                completion,
                add_special_tokens=False,
            ).input_ids
        )

        backbone = self.backbone_text(completion)
        backbone_tokens = len(
            self.policy_tokenizer(
                backbone,
                add_special_tokens=False,
            ).input_ids
        )

        denominator = max(int(source_tokens), 1)
        return {
            "backbone_tokens": int(backbone_tokens),
            "full_completion_tokens": int(full_tokens),
            "compression_ratio": backbone_tokens / denominator,
            "full_compression_ratio": full_tokens / denominator,
        }

    # def budget_reward_from_ratio(self, ratio: float) -> float:
    #     excess = max(0.0, ratio - self.target_ratio)
    #     penalty = excess / max(self.target_ratio, 1e-8)
    #     return -penalty
    def budget_reward_from_ratio(self, ratio: float) -> float:
        half_width = 0.05

        low = max(0.0, self.target_ratio - half_width)
        high = min(1.0, self.target_ratio + half_width)

        if ratio < low:
            return -((low - ratio) / max(low, 1e-8))

        if ratio > high:
            return -((ratio - high) / max(1.0 - high, 1e-8))

        return 0.0

    @staticmethod
    def _parse_completion(completion: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(completion.strip())
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        if not isinstance(obj.get("summary"), str):
            return None
        if not isinstance(obj.get("link"), dict):
            return None
        return obj

    def backbone_text(self, completion: str) -> str:
        obj = self._parse_completion(completion)
        if obj is None:
            return completion
        return obj["summary"]

    def reader_view(
        self,
        completion: str,
        omit_cite: Optional[str] = None,
        backbone_only: bool = False,
    ) -> str:
        """Render Reader input; optionally keep only the summary/backbone."""
        obj = self._parse_completion(completion)
        if obj is None:
            return completion

        parts = [f"Summary:\n{obj['summary']}"]
        if backbone_only:
            return parts[0]

        blocks = []
        for key, value in obj["link"].items():
            if key == omit_cite:
                continue
            if isinstance(value, str):
                blocks.append(f"[{key}] {value}")

        if blocks:
            parts.append("Expandable information:\n" + "\n".join(blocks))
        return "\n\n".join(parts)

    def cite_keys(self, completion: str) -> List[str]:
        obj = self._parse_completion(completion)
        if obj is None:
            return []
        return [
            str(key)
            for key, value in obj["link"].items()
            if re.fullmatch(r"cite_\d+", str(key)) and isinstance(value, str)
        ]

    def _render_judge_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            return self.reader_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.reader_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _classify_next_token(
        self,
        prompts: List[str],
        candidate_token_ids: List[int],
        return_scores: bool = False,
    ):
        predictions: List[int] = []
        score_rows: List[Dict[str, Any]] = []

        for start in range(0, len(prompts), self.reader_batch_size):
            batch_prompts = prompts[start:start + self.reader_batch_size]

            inputs = self.reader_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.reader_device)

            with torch.inference_mode():
                outputs = self.reader(
                    **inputs,
                    use_cache=False,
                    logits_to_keep=1,
                )

            next_logits = outputs.logits[:, -1, :].float()
            candidate_ids = torch.tensor(
                candidate_token_ids,
                device=self.reader_device,
            )
            candidate_logits = next_logits.index_select(
                dim=-1,
                index=candidate_ids,
            )

            predictions.extend(candidate_logits.argmax(dim=-1).tolist())

            if return_scores:
                candidate_probs = torch.softmax(candidate_logits, dim=-1)
                log_norm = torch.logsumexp(next_logits, dim=-1)
                candidate_mass = torch.exp(
                    torch.logsumexp(candidate_logits, dim=-1) - log_norm
                )
                top_logits, top_ids = torch.topk(next_logits, k=5, dim=-1)
                top_probs = torch.exp(top_logits - log_norm.unsqueeze(-1))

                probs_cpu = candidate_probs.detach().cpu().tolist()
                mass_cpu = candidate_mass.detach().cpu().tolist()
                top_ids_cpu = top_ids.detach().cpu().tolist()
                top_probs_cpu = top_probs.detach().cpu().tolist()

                for probs, mass, ids, probs_top in zip(
                    probs_cpu, mass_cpu, top_ids_cpu, top_probs_cpu
                ):
                    row = {
                        "candidate_probs": [float(x) for x in probs],
                        "candidate_mass": float(mass),
                    }
                    if mass < self.CANDIDATE_MASS_THRESHOLD:
                        row["top_tokens"] = [
                            {
                                "token_id": int(token_id),
                                "decoded": self.reader_tokenizer.decode([token_id]),
                                "prob": float(prob),
                            }
                            for token_id, prob in zip(ids, probs_top)
                        ]
                    score_rows.append(row)

        if return_scores:
            return predictions, score_rows
        return predictions

    def _single_token_ids(self, candidates: List[str]) -> List[int]:
        token_ids = []

        for candidate in candidates:
            ids = self.reader_tokenizer(
                candidate,
                add_special_tokens=False,
            ).input_ids

            if len(ids) != 1:
                raise ValueError(
                    f"Reader candidate {candidate!r} is not a single token: {ids}"
                )

            token_ids.append(ids[0])

        return token_ids

    def _candidate_meta(self, labels: List[str], token_ids: List[int]) -> List[Dict[str, Any]]:
        return [
            {
                "label": label,
                "token_id": int(token_id),
                "decoded": self.reader_tokenizer.decode([token_id]),
            }
            for label, token_id in zip(labels, token_ids)
        ]
