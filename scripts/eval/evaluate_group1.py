#!/usr/bin/env python3
"""
Group-1 QA evaluator for LoCoMo and LongMemEval.

Primary metrics
---------------
LoCoMo:
  - Main semantic correctness accuracy on categories 1-4.
  - Category-5 adversarial / abstention accuracy reported separately.
  - Performance Retention = condition main accuracy / full-context main accuracy.
  - Benchmark-native LoCoMo F1-style score as an appendix metric.

LongMemEval:
  - LLM-as-judge QA accuracy using the official benchmark's task-specific
    correctness semantics, implemented with a replaceable local BinaryJudge.
  - Abstention questions are included in the primary overall accuracy.
  - Performance Retention = condition accuracy / full-context accuracy.

The judge model is never hard-coded. Swap it with --judge-model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from judge import BinaryJudge


LOCOMO_CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict benchmark evaluator. Decide whether the model response "
    "satisfies the evaluation rule. Return label 1 for correct and label 0 "
    "for incorrect. Do not produce explanations; the scoring code evaluates "
    "only the fixed candidate labels 0 and 1."
)

LOCOMO_PROMPT_VERSION = "locomo_semantic_binary_v1"
LONGMEMEVAL_PROMPT_VERSION = "longmemeval_official_semantics_binary_v1"


# ======================================================================
# CLI / I/O
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        required=True,
        choices=["locomo", "longmemeval"],
    )
    p.add_argument(
        "--eval-meta",
        required=True,
        help="Processed eval_meta.jsonl.",
    )
    p.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="CONDITION=PATH",
        help=(
            "Prediction JSONL. Repeat for each condition, e.g. "
            "--prediction full_context=... --prediction stage1=..."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for per_question.jsonl, metrics.json, table.csv.",
    )

    # Replaceable judge backend.
    p.add_argument("--judge-model", required=True)
    p.add_argument(
        "--judge-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    p.add_argument(
        "--judge-device-map",
        default="auto",
        help="'auto', 'cpu', or an explicit device such as cuda:0.",
    )
    p.add_argument(
        "--judge-attn-implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        default="sdpa",
    )
    p.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    p.set_defaults(trust_remote_code=True)

    p.add_argument(
        "--judge-batch-size",
        type=int,
        default=8,
        help="Batch size for single-token 0/1 judge scoring.",
    )
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help=(
            "Number of paired bootstrap resamples for Stage-1 vs Stage-2 "
            "confidence intervals. Set 0 to disable."
        ),
    )
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Deterministic seed for paired bootstrap analysis.",
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Smoke-test limit. The same gold-question subset is used for every "
            "condition."
        ),
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Evaluate available predictions instead of failing on missing IDs.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing evaluation outputs and start over.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate joins without loading the judge model.",
    )

    args = p.parse_args()
    if args.judge_batch_size < 1:
        p.error("--judge-batch-size must be >= 1")
    if args.bootstrap_samples < 0:
        p.error("--bootstrap-samples must be >= 0")
    return args


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
    return rows


def parse_prediction_args(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --prediction '{item}'. Expected CONDITION=PATH."
            )
        condition, path = item.split("=", 1)
        condition = condition.strip()
        path = path.strip()
        if not condition or not path:
            raise ValueError(
                f"Invalid --prediction '{item}'. Expected CONDITION=PATH."
            )
        if condition in out:
            raise ValueError(f"Duplicate prediction condition: {condition}")
        out[condition] = Path(path)
    return out


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def textify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return ", ".join(textify(v) for v in x)
    if isinstance(x, (int, float, bool)):
        return str(x)
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


# ======================================================================
# Metadata normalization
# ======================================================================

def load_gold_questions(
    dataset: str,
    meta_path: Path,
) -> List[Dict[str, Any]]:
    rows = read_jsonl(meta_path)
    gold: List[Dict[str, Any]] = []

    if dataset == "locomo":
        for row in rows:
            source_id = str(row["id"])
            questions = row.get("questions")
            if not isinstance(questions, list):
                raise ValueError(f"{source_id}: LoCoMo row has no questions list")

            for idx, q in enumerate(questions):
                if not isinstance(q, dict):
                    raise ValueError(f"{source_id}: invalid question at index {idx}")

                qid = str(q.get("question_id", f"{source_id}::q{idx:04d}"))
                category = int(q["category"])
                if category not in LOCOMO_CATEGORY_NAMES:
                    raise ValueError(f"{qid}: unknown LoCoMo category {category}")

                gold.append({
                    "dataset": "locomo",
                    "source_id": source_id,
                    "question_id": qid,
                    "question": textify(q.get("question")),
                    "gold_answer": textify(q.get("answer")),
                    "adversarial_answer": textify(q.get("adversarial_answer")),
                    "category": category,
                    "category_name": LOCOMO_CATEGORY_NAMES[category],
                    "question_type": None,
                    "is_abstention": category == 5,
                })

    elif dataset == "longmemeval":
        for row in rows:
            source_id = str(row["id"])
            qid = str(row["question_id"])

            gold.append({
                "dataset": "longmemeval",
                "source_id": source_id,
                "question_id": qid,
                "question": textify(row.get("question")),
                "gold_answer": textify(row.get("answer")),
                "adversarial_answer": "",
                "category": None,
                "category_name": None,
                "question_type": row.get("question_type"),
                "is_abstention": bool(
                    row.get("is_abstention", qid.endswith("_abs"))
                ),
            })

    else:
        raise ValueError(dataset)

    seen = set()
    for q in gold:
        key = (q["source_id"], q["question_id"])
        if key in seen:
            raise ValueError(f"Duplicate gold question key: {key}")
        seen.add(key)

        if not q["question"]:
            raise ValueError(f"{key}: empty question")

        # LoCoMo category 5 is adversarial / unanswerable. Its correctness
        # is judged by whether the model abstains, so a normal gold answer is
        # not required (some rows leave `answer` empty).
        if q["dataset"] == "locomo" and q["category"] == 5:
            continue

        # Ordinary LoCoMo QA and all LongMemEval rows require a reference
        # answer/rubric for semantic correctness evaluation.
        if not q["gold_answer"]:
            raise ValueError(f"{key}: empty gold answer/rubric")

    return gold


def load_predictions(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    rows = read_jsonl(path)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in rows:
        if "source_id" not in row or "question_id" not in row:
            raise ValueError(
                f"{path}: prediction row missing source_id/question_id"
            )

        key = (str(row["source_id"]), str(row["question_id"]))
        if key in out:
            raise ValueError(f"{path}: duplicate prediction key {key}")

        prediction = row.get("prediction")
        if prediction is None:
            prediction = row.get("hypothesis")
        if prediction is None:
            raise ValueError(f"{path}: {key} has no prediction/hypothesis")

        out[key] = {
            **row,
            "prediction": textify(prediction),
        }

    return out


# ======================================================================
# Judge prompt construction
# ======================================================================

def build_locomo_prompt(
    q: Dict[str, Any],
    prediction: str,
) -> Tuple[str, str, str]:
    category = q["category"]

    if category == 5:
        user = f"""Evaluation rule:
This is an adversarial/unanswerable LoCoMo question. Label 1 only if the model response correctly recognizes that the requested information is unavailable, not mentioned, or cannot be determined from the conversation. A concise refusal or statement of insufficient information is correct. Label 0 if the response asserts a concrete unsupported answer or otherwise behaves as if the question were answerable.

Question:
{q["question"]}

Model response:
{prediction}

Return the correctness label."""
        return JUDGE_SYSTEM_PROMPT, user, LOCOMO_PROMPT_VERSION

    user = f"""Evaluation rule:
Compare the model response with the reference answer. Label 1 when the response is semantically equivalent to the reference answer or clearly contains all information required by it. Paraphrases are acceptable. Additional information is acceptable unless it contradicts the required answer. If only a subset of the required information is provided, or any required fact is wrong, label 0.

Question:
{q["question"]}

Reference answer:
{q["gold_answer"]}

Model response:
{prediction}

Return the correctness label."""
    return JUDGE_SYSTEM_PROMPT, user, LOCOMO_PROMPT_VERSION


def build_longmemeval_prompt(
    q: Dict[str, Any],
    prediction: str,
) -> Tuple[str, str, str]:
    question = q["question"]
    answer = q["gold_answer"]

    if q["is_abstention"]:
        user = f"""Evaluation rule:
The question is unanswerable from the conversation history. The reference text below explains the expected situation. Label 1 if the model response correctly identifies that the requested information is unavailable, incomplete, or cannot be determined. It may mention other available information, but it must not pretend the missing requested information is known. Otherwise label 0.

Question:
{question}

Reference explanation:
{answer}

Model response:
{prediction}

Return the correctness label."""
        return JUDGE_SYSTEM_PROMPT, user, LONGMEMEVAL_PROMPT_VERSION

    task = q["question_type"]

    if task in {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    }:
        rule = (
            "Label 1 if the response contains the correct answer, is semantically "
            "equivalent to it, or supplies all intermediate information needed to "
            "derive it. A response that gives only a subset of the information "
            "required by the reference answer is incorrect."
        )
        answer_header = "Reference answer"

    elif task == "temporal-reasoning":
        rule = (
            "Use ordinary semantic correctness, but do not penalize an off-by-one "
            "error when the requested result is a count of elapsed days, weeks, "
            "months, or a similar time interval. Other substantive temporal errors "
            "remain incorrect."
        )
        answer_header = "Reference answer"

    elif task == "knowledge-update":
        rule = (
            "Label 1 if the response includes the required updated/current answer. "
            "It may also mention previous outdated information as long as the "
            "correct updated answer is clearly included. Label 0 if the required "
            "update is absent or wrong."
        )
        answer_header = "Reference answer"

    elif task == "single-session-preference":
        rule = (
            "The reference text is a rubric for a desired personalized response, "
            "not necessarily a literal target string. Label 1 when the response "
            "correctly recalls and uses the user's personal information in a way "
            "that satisfies the desired personalization. It need not cover every "
            "rubric point. Label 0 when personalization is missing, based on wrong "
            "personal information, or conflicts with the rubric."
        )
        answer_header = "Personalization rubric"

    else:
        raise ValueError(
            f"{q['question_id']}: unsupported LongMemEval "
            f"question_type={task!r}"
        )

    user = f"""Evaluation rule:
{rule}

Question:
{question}

{answer_header}:
{answer}

Model response:
{prediction}

Return the correctness label."""
    return JUDGE_SYSTEM_PROMPT, user, LONGMEMEVAL_PROMPT_VERSION


def build_judge_prompt(
    dataset: str,
    q: Dict[str, Any],
    prediction: str,
) -> Tuple[str, str, str]:
    if dataset == "locomo":
        return build_locomo_prompt(q, prediction)
    if dataset == "longmemeval":
        return build_longmemeval_prompt(q, prediction)
    raise ValueError(dataset)


# ======================================================================
# LoCoMo benchmark-native appendix metric
# ======================================================================

def _porter():
    try:
        from nltk.stem import PorterStemmer
    except ImportError as e:
        raise RuntimeError(
            "LoCoMo native F1 requires nltk. Install it with: pip install nltk"
        ) from e
    return PorterStemmer()


def locomo_normalize_answer(s: str) -> str:
    s = str(s).replace(",", "")

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return " ".join(
        remove_articles(remove_punc(s.lower())).split()
    )


def locomo_f1_score(
    prediction: str,
    ground_truth: str,
    stemmer,
) -> float:
    pred_tokens = [
        stemmer.stem(w)
        for w in locomo_normalize_answer(prediction).split()
    ]
    gold_tokens = [
        stemmer.stem(w)
        for w in locomo_normalize_answer(ground_truth).split()
    ]

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def locomo_multi_answer_f1(
    prediction: str,
    ground_truth: str,
    stemmer,
) -> float:
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]

    if not ground_truths:
        return 0.0

    vals = []
    for gt in ground_truths:
        vals.append(
            max(
                locomo_f1_score(pred, gt, stemmer)
                for pred in predictions
            )
        )
    return sum(vals) / len(vals)


def locomo_native_score(
    prediction: str,
    answer: str,
    category: int,
    stemmer,
) -> float:
    # Mirrors the public LoCoMo evaluation behavior.
    if category == 3:
        answer = answer.split(";")[0].strip()

    if category in {2, 3, 4}:
        return locomo_f1_score(prediction, answer, stemmer)

    if category == 1:
        return locomo_multi_answer_f1(prediction, answer, stemmer)

    if category == 5:
        p = prediction.lower()
        return float(
            "no information available" in p
            or "not mentioned" in p
        )

    raise ValueError(f"Unknown LoCoMo category: {category}")


# ======================================================================
# Resume / validation
# ======================================================================

def make_run_config(
    args,
    prediction_paths: Dict[str, Path],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "eval_meta": str(Path(args.eval_meta).resolve()),
        "predictions": {
            k: str(v.resolve())
            for k, v in sorted(prediction_paths.items())
        },
        "judge_request": {
            "model": args.judge_model,
            "dtype": args.judge_dtype,
            "device_map": args.judge_device_map,
            "attn_implementation": args.judge_attn_implementation,
            "trust_remote_code": args.trust_remote_code,
        },
        "judge_protocol": {
            "method": "binary_conditional_logprob",
            "false_label": "0",
            "true_label": "1",
            "locomo_prompt_version": LOCOMO_PROMPT_VERSION,
            "longmemeval_prompt_version": LONGMEMEVAL_PROMPT_VERSION,
        },
        "max_questions": args.max_questions,
        "allow_missing": args.allow_missing,
    }


def prepare_output_dir(
    output_dir: Path,
    config: Dict[str, Any],
    overwrite: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "run_config.json"
    perq_path = output_dir / "per_question.jsonl"
    metrics_path = output_dir / "metrics.json"
    table_path = output_dir / "table.csv"

    if overwrite:
        for p in [config_path, perq_path, metrics_path, table_path]:
            if p.exists():
                p.unlink()

    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        # Ignore runtime metadata inserted after the Judge is loaded.
        old_compare = {k: v for k, v in old.items() if k != "judge_runtime"}
        if stable_json_hash(old_compare) != stable_json_hash(config):
            raise RuntimeError(
                "Existing output directory uses a different evaluation "
                "configuration. Use a new --output-dir or --overwrite."
            )
    else:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return perq_path, metrics_path, table_path, config_path


def load_completed(
    perq_path: Path,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out = {}

    if not perq_path.exists():
        return out

    for row in read_jsonl(perq_path):
        key = (
            str(row["condition"]),
            str(row["source_id"]),
            str(row["question_id"]),
        )
        if key in out:
            raise RuntimeError(f"Duplicate existing evaluation result: {key}")
        out[key] = row

    return out


def validate_prediction_coverage(
    gold: Sequence[Dict[str, Any]],
    predictions_by_condition: Dict[
        str,
        Dict[Tuple[str, str], Dict[str, Any]],
    ],
    *,
    allow_missing: bool,
):
    gold_keys = {
        (q["source_id"], q["question_id"])
        for q in gold
    }

    for condition, preds in predictions_by_condition.items():
        pred_keys = set(preds)
        missing = sorted(gold_keys - pred_keys)
        extras = sorted(pred_keys - gold_keys)

        if missing and not allow_missing:
            raise RuntimeError(
                f"{condition}: missing {len(missing)} required predictions. "
                f"First missing: {missing[:3]}"
            )

        if extras:
            print(
                f"WARNING: {condition}: {len(extras)} prediction rows do not "
                f"match the selected evaluation set and will be ignored. "
                f"First extras: {extras[:3]}"
            )


# ======================================================================
# Aggregation
# ======================================================================

def mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(sum(values) / len(values))


def safe_ratio(
    num: Optional[float],
    den: Optional[float],
) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return float(num / den)


def aggregate_locomo(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    main = [
        r for r in rows
        if int(r["category"]) in {1, 2, 3, 4}
    ]
    adv = [
        r for r in rows
        if int(r["category"]) == 5
    ]

    by_cat: Dict[str, Dict[str, Any]] = {}
    for cat in [1, 2, 3, 4, 5]:
        subset = [
            r for r in rows
            if int(r["category"]) == cat
        ]
        by_cat[str(cat)] = {
            "name": LOCOMO_CATEGORY_NAMES[cat],
            "n": len(subset),
            "judge_accuracy": mean([
                r["judge_label"] for r in subset
            ]),
            "native_score": mean([
                r["native_score"] for r in subset
                if r.get("native_score") is not None
            ]),
        }

    return {
        "n": len(rows),
        "main_n_cat1_4": len(main),
        "main_accuracy": mean([
            r["judge_label"] for r in main
        ]),
        "adversarial_n_cat5": len(adv),
        "adversarial_accuracy": mean([
            r["judge_label"] for r in adv
        ]),
        "all_judge_accuracy": mean([
            r["judge_label"] for r in rows
        ]),
        "native_main_score_cat1_4": mean([
            r["native_score"] for r in main
            if r.get("native_score") is not None
        ]),
        "native_all_score": mean([
            r["native_score"] for r in rows
            if r.get("native_score") is not None
        ]),
        "by_category": by_cat,
    }


def aggregate_longmemeval(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, Any]] = {}

    types = sorted({
        str(r["question_type"])
        for r in rows
    })

    for t in types:
        subset = [
            r for r in rows
            if str(r["question_type"]) == t
        ]
        by_type[t] = {
            "n": len(subset),
            "accuracy": mean([
                r["judge_label"] for r in subset
            ]),
        }

    abs_rows = [
        r for r in rows
        if r["is_abstention"]
    ]
    non_abs = [
        r for r in rows
        if not r["is_abstention"]
    ]

    return {
        "n": len(rows),
        "main_accuracy": mean([
            r["judge_label"] for r in rows
        ]),
        "non_abstention_n": len(non_abs),
        "non_abstention_accuracy": mean([
            r["judge_label"] for r in non_abs
        ]),
        "abstention_n": len(abs_rows),
        "abstention_accuracy": mean([
            r["judge_label"] for r in abs_rows
        ]),
        "by_question_type": by_type,
    }


def aggregate_all(
    dataset: str,
    rows: Sequence[Dict[str, Any]],
    conditions: Sequence[str],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    for condition in conditions:
        subset = [
            r for r in rows
            if r["condition"] == condition
        ]

        if dataset == "locomo":
            metrics[condition] = aggregate_locomo(subset)
        else:
            metrics[condition] = aggregate_longmemeval(subset)

    baseline = metrics.get(
        "full_context",
        {},
    ).get("main_accuracy")

    for condition, m in metrics.items():
        if condition == "full_context":
            m["performance_retention"] = (
                1.0 if baseline is not None else None
            )
        else:
            m["performance_retention"] = safe_ratio(
                m.get("main_accuracy"),
                baseline,
            )

    return metrics



# ======================================================================
# Paired system comparison / Judge diagnostics
# ======================================================================

def _main_metric_row(dataset: str, row: Dict[str, Any]) -> bool:
    if dataset == "locomo":
        return int(row["category"]) in {1, 2, 3, 4}
    return True


def _condition_main_map(
    dataset: str,
    rows: Sequence[Dict[str, Any]],
    condition: str,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out = {}
    for r in rows:
        if r["condition"] != condition:
            continue
        if not _main_metric_row(dataset, r):
            continue
        key = (str(r["source_id"]), str(r["question_id"]))
        if key in out:
            raise RuntimeError(
                f"Duplicate result for condition={condition}, key={key}"
            )
        out[key] = r
    return out


def transition_matrix(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
) -> Dict[str, Any]:
    if len(labels_a) != len(labels_b):
        raise ValueError("Paired label arrays must have equal length.")

    n00 = n01 = n10 = n11 = 0
    for a, b in zip(labels_a, labels_b):
        a = int(a)
        b = int(b)
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 1 and b == 1:
            n11 += 1
        else:
            raise ValueError(f"Labels must be binary, got {a}, {b}")

    n = len(labels_a)
    return {
        "n": n,
        "0_to_0": n00,
        "0_to_1": n01,
        "1_to_0": n10,
        "1_to_1": n11,
        "stage2_wins": n01,
        "stage2_losses": n10,
        "ties": n00 + n11,
        "net_correct_gain": n01 - n10,
        "accuracy_delta_stage2_minus_stage1": (
            (n01 - n10) / n if n else None
        ),
    }


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def exact_mcnemar_test(
    n01: int,
    n10: int,
) -> Dict[str, Any]:
    """
    Exact two-sided McNemar test.

    Under H0, among discordant pairs, Stage-2 wins are Binomial(n, 0.5).
    The two-sided exact p-value is 2 * lower-tail probability at
    min(n01, n10), capped at 1.
    """
    discordant = int(n01) + int(n10)
    if discordant == 0:
        return {
            "discordant_pairs": 0,
            "statistic_abs_difference": 0,
            "p_value_exact_two_sided": 1.0,
            "significant_at_0_05": False,
        }

    m = min(int(n01), int(n10))
    log2 = math.log(2.0)
    log_probs = []
    for k in range(m + 1):
        log_p = (
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            - discordant * log2
        )
        log_probs.append(log_p)

    lower_tail = math.exp(_logsumexp(log_probs))
    p = min(1.0, 2.0 * lower_tail)

    return {
        "discordant_pairs": discordant,
        "statistic_abs_difference": abs(int(n01) - int(n10)),
        "p_value_exact_two_sided": float(p),
        "significant_at_0_05": bool(p < 0.05),
    }


def percentile(
    values: Sequence[float],
    q: float,
) -> Optional[float]:
    if not values:
        return None
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")

    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]

    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]

    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def paired_bootstrap_stage1_stage2(
    stage1: Sequence[int],
    stage2: Sequence[int],
    full_context: Optional[Sequence[int]],
    *,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    if len(stage1) != len(stage2):
        raise ValueError("Stage-1 and Stage-2 must be paired.")
    if full_context is not None and len(full_context) != len(stage1):
        raise ValueError("Full-context labels must use the same paired IDs.")

    n = len(stage1)
    if n == 0:
        return {
            "samples_requested": samples,
            "samples_used": 0,
            "seed": seed,
            "accuracy_delta": None,
            "retention_delta": None,
        }

    acc1 = sum(stage1) / n
    acc2 = sum(stage2) / n
    point_acc_delta = acc2 - acc1

    full_acc = None
    point_ret_delta = None
    if full_context is not None:
        full_acc = sum(full_context) / n
        if full_acc > 0:
            point_ret_delta = (acc2 / full_acc) - (acc1 / full_acc)

    if samples == 0:
        return {
            "samples_requested": 0,
            "samples_used": 0,
            "seed": seed,
            "accuracy_delta": {
                "point_estimate": point_acc_delta,
                "ci95_percentile": None,
            },
            "retention_delta": {
                "point_estimate": point_ret_delta,
                "ci95_percentile": None,
            },
        }

    rng = random.Random(seed)
    acc_deltas = []
    ret_deltas = []

    for _ in range(samples):
        s1 = 0
        s2 = 0
        sf = 0

        for _j in range(n):
            i = rng.randrange(n)
            s1 += int(stage1[i])
            s2 += int(stage2[i])
            if full_context is not None:
                sf += int(full_context[i])

        b1 = s1 / n
        b2 = s2 / n
        acc_deltas.append(b2 - b1)

        if full_context is not None:
            bf = sf / n
            if bf > 0:
                ret_deltas.append((b2 / bf) - (b1 / bf))

    return {
        "samples_requested": samples,
        "samples_used": len(acc_deltas),
        "retention_samples_used": len(ret_deltas),
        "seed": seed,
        "accuracy_delta": {
            "point_estimate": point_acc_delta,
            "ci95_percentile": [
                percentile(acc_deltas, 0.025),
                percentile(acc_deltas, 0.975),
            ],
            "bootstrap_probability_delta_gt_0": (
                sum(x > 0 for x in acc_deltas) / len(acc_deltas)
                if acc_deltas else None
            ),
        },
        "retention_delta": {
            "point_estimate": point_ret_delta,
            "ci95_percentile": (
                [
                    percentile(ret_deltas, 0.025),
                    percentile(ret_deltas, 0.975),
                ]
                if ret_deltas else None
            ),
            "bootstrap_probability_delta_gt_0": (
                sum(x > 0 for x in ret_deltas) / len(ret_deltas)
                if ret_deltas else None
            ),
        },
        "full_context_accuracy_on_paired_subset": full_acc,
    }


def judge_candidate_mass_summary(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    vals = [
        float(r["judge_candidate_mass"])
        for r in rows
        if r.get("judge_candidate_mass") is not None
    ]
    if not vals:
        return {
            "n_with_candidate_mass": 0,
            "note": (
                "candidate_mass is unavailable when the Judge uses "
                "multi-token candidate-sequence scoring"
            ),
        }

    return {
        "n_with_candidate_mass": len(vals),
        "mean": mean(vals),
        "p10": percentile(vals, 0.10),
        "median": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "min": min(vals),
        "max": max(vals),
        "fraction_lt_0_5": sum(x < 0.5 for x in vals) / len(vals),
        "fraction_lt_0_9": sum(x < 0.9 for x in vals) / len(vals),
    }


def paired_breakdown(
    dataset: str,
    paired_rows: Sequence[
        Tuple[Dict[str, Any], Dict[str, Any]]
    ],
) -> Dict[str, Any]:
    """
    Transition matrices by LoCoMo category or LongMemEval question type.
    These are diagnostic, not independently significance-tested.
    """
    groups: Dict[str, List[Tuple[int, int]]] = {}

    for r1, r2 in paired_rows:
        if dataset == "locomo":
            group = (
                f"category_{int(r1['category'])}_"
                f"{r1.get('category_name')}"
            )
        else:
            group = str(r1.get("question_type"))

        groups.setdefault(group, []).append(
            (int(r1["judge_label"]), int(r2["judge_label"]))
        )

    out = {}
    for group, pairs in sorted(groups.items()):
        a = [x[0] for x in pairs]
        b = [x[1] for x in pairs]
        out[group] = transition_matrix(a, b)

    return out


def stage1_stage2_analysis(
    dataset: str,
    rows: Sequence[Dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Optional[Dict[str, Any]]:
    conditions = {str(r["condition"]) for r in rows}
    if "stage1" not in conditions or "stage2" not in conditions:
        return None

    m1 = _condition_main_map(dataset, rows, "stage1")
    m2 = _condition_main_map(dataset, rows, "stage2")
    mf = (
        _condition_main_map(dataset, rows, "full_context")
        if "full_context" in conditions
        else {}
    )

    common = sorted(set(m1) & set(m2))
    if not common:
        return None

    labels1 = [int(m1[k]["judge_label"]) for k in common]
    labels2 = [int(m2[k]["judge_label"]) for k in common]

    full_labels = None
    bootstrap_keys = common
    if mf:
        bootstrap_keys = sorted(set(common) & set(mf))
        if bootstrap_keys:
            labels1_boot = [int(m1[k]["judge_label"]) for k in bootstrap_keys]
            labels2_boot = [int(m2[k]["judge_label"]) for k in bootstrap_keys]
            full_labels = [int(mf[k]["judge_label"]) for k in bootstrap_keys]
        else:
            labels1_boot = labels1
            labels2_boot = labels2
    else:
        labels1_boot = labels1
        labels2_boot = labels2

    tm = transition_matrix(labels1, labels2)

    paired_rows = [(m1[k], m2[k]) for k in common]

    return {
        "scope": (
            "LoCoMo categories 1-4"
            if dataset == "locomo"
            else "LongMemEval primary overall questions"
        ),
        "paired_questions": len(common),
        "transition_matrix": tm,
        "mcnemar_exact": exact_mcnemar_test(
            tm["0_to_1"],
            tm["1_to_0"],
        ),
        "paired_bootstrap": paired_bootstrap_stage1_stage2(
            labels1_boot,
            labels2_boot,
            full_labels,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "breakdown": paired_breakdown(
            dataset,
            paired_rows,
        ),
    }


def all_judge_diagnostics(
    rows: Sequence[Dict[str, Any]],
    conditions: Sequence[str],
) -> Dict[str, Any]:
    out = {}
    for condition in conditions:
        subset = [
            r for r in rows
            if r["condition"] == condition
        ]
        out[condition] = {
            "all_questions": judge_candidate_mass_summary(subset),
            "main_metric_questions": judge_candidate_mass_summary([
                r for r in subset
                if _main_metric_row(
                    str(r["dataset"]),
                    r,
                )
            ]),
        }
    return out


def write_table(
    path: Path,
    dataset: str,
    metrics: Dict[str, Any],
    condition_order: Sequence[str],
):
    if dataset == "locomo":
        fields = [
            "condition",
            "n",
            "main_accuracy_cat1_4",
            "performance_retention",
            "adversarial_accuracy_cat5",
            "all_judge_accuracy",
            "native_main_score_cat1_4",
            "native_all_score",
        ]
    else:
        fields = [
            "condition",
            "n",
            "main_accuracy",
            "performance_retention",
            "non_abstention_accuracy",
            "abstention_accuracy",
        ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for condition in condition_order:
            m = metrics[condition]

            if dataset == "locomo":
                row = {
                    "condition": condition,
                    "n": m["n"],
                    "main_accuracy_cat1_4": m["main_accuracy"],
                    "performance_retention": m["performance_retention"],
                    "adversarial_accuracy_cat5": m["adversarial_accuracy"],
                    "all_judge_accuracy": m["all_judge_accuracy"],
                    "native_main_score_cat1_4": (
                        m["native_main_score_cat1_4"]
                    ),
                    "native_all_score": m["native_all_score"],
                }
            else:
                row = {
                    "condition": condition,
                    "n": m["n"],
                    "main_accuracy": m["main_accuracy"],
                    "performance_retention": m["performance_retention"],
                    "non_abstention_accuracy": (
                        m["non_abstention_accuracy"]
                    ),
                    "abstention_accuracy": m["abstention_accuracy"],
                }

            writer.writerow(row)


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    meta_path = Path(args.eval_meta)
    prediction_paths = parse_prediction_args(
        args.prediction
    )
    condition_order = list(
        prediction_paths.keys()
    )

    gold = load_gold_questions(
        args.dataset,
        meta_path,
    )

    if args.max_questions is not None:
        gold = gold[:args.max_questions]

    predictions_by_condition = {
        condition: load_predictions(path)
        for condition, path in prediction_paths.items()
    }

    validate_prediction_coverage(
        gold,
        predictions_by_condition,
        allow_missing=args.allow_missing,
    )

    print("=" * 78)
    print("GROUP-1 QA EVALUATOR")
    print("=" * 78)
    print(f"dataset          : {args.dataset}")
    print(f"eval meta        : {meta_path}")
    print(f"questions        : {len(gold)}")
    print(f"conditions       : {', '.join(condition_order)}")
    for condition, path in prediction_paths.items():
        print(f"  {condition:<15}: {path}")
    print(f"judge model      : {args.judge_model}")
    print(f"judge dtype      : {args.judge_dtype}")
    print(f"judge device map : {args.judge_device_map}")
    print(f"judge attention  : {args.judge_attn_implementation}")
    print(f"judge batch size : {args.judge_batch_size}")
    print("judge generation : NEVER")
    print("=" * 78)

    if args.validate_only:
        print("Validation passed. Judge model was not loaded.")
        return

    output_dir = Path(args.output_dir)

    run_config = make_run_config(
        args,
        prediction_paths,
    )

    (
        perq_path,
        metrics_path,
        table_path,
        config_path,
    ) = prepare_output_dir(
        output_dir,
        run_config,
        args.overwrite,
    )

    completed = load_completed(
        perq_path
    )

    jobs = []

    for condition in condition_order:
        preds = predictions_by_condition[
            condition
        ]

        for q in gold:
            key = (
                q["source_id"],
                q["question_id"],
            )

            if key not in preds:
                if args.allow_missing:
                    continue
                raise AssertionError(
                    "Prediction coverage should already have been validated."
                )

            eval_key = (
                condition,
                q["source_id"],
                q["question_id"],
            )
            if eval_key in completed:
                continue

            pred = preds[key]["prediction"]

            (
                system_prompt,
                user_prompt,
                prompt_version,
            ) = build_judge_prompt(
                args.dataset,
                q,
                pred,
            )

            jobs.append({
                "condition": condition,
                "gold": q,
                "prediction": pred,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "prompt_version": prompt_version,
            })

    print(f"already judged   : {len(completed)}")
    print(f"pending judgments: {len(jobs)}")

    judge = BinaryJudge(
        args.judge_model,
        dtype=args.judge_dtype,
        device_map=args.judge_device_map,
        attn_implementation=args.judge_attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    run_config["judge_runtime"] = (
        judge.metadata()
    )
    config_path.write_text(
        json.dumps(
            run_config,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    stemmer = (
        _porter()
        if args.dataset == "locomo"
        else None
    )

    mode = (
        "a"
        if perq_path.exists()
        else "w"
    )

    with perq_path.open(
        mode,
        encoding="utf-8",
    ) as fout:
        for start in tqdm(
            range(
                0,
                len(jobs),
                args.judge_batch_size,
            ),
            desc="Judging",
        ):
            batch_jobs = jobs[
                start:start + args.judge_batch_size
            ]

            judge_inputs = [
                {
                    "system_prompt": job["system_prompt"],
                    "user_prompt": job["user_prompt"],
                }
                for job in batch_jobs
            ]

            results = judge.score_batch(
                judge_inputs
            )

            if len(results) != len(batch_jobs):
                raise RuntimeError(
                    "Judge batch result length mismatch."
                )

            for job, result in zip(
                batch_jobs,
                results,
            ):
                q = job["gold"]

                native_score = None
                if args.dataset == "locomo":
                    native_score = (
                        locomo_native_score(
                            job["prediction"],
                            q["gold_answer"],
                            int(q["category"]),
                            stemmer,
                        )
                    )

                record = {
                    "dataset": args.dataset,
                    "condition": job["condition"],
                    "source_id": q["source_id"],
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "gold_answer": q["gold_answer"],
                    "prediction": job["prediction"],
                    "category": q["category"],
                    "category_name": q["category_name"],
                    "question_type": q["question_type"],
                    "is_abstention": q["is_abstention"],
                    # Audit only; never used as direct category-5 gold.
                    "adversarial_answer": q["adversarial_answer"],
                    "judge_label": result.label,
                    "judge_p0": result.p0,
                    "judge_p1": result.p1,
                    "judge_logp0": result.logp0,
                    "judge_logp1": result.logp1,
                    "judge_candidate_mass": result.candidate_mass,
                    "judge_scoring_mode": result.scoring_mode,
                    "judge_prompt_version": job["prompt_version"],
                    "native_score": native_score,
                }

                fout.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    ) + "\n"
                )
                fout.flush()

    all_rows = read_jsonl(
        perq_path
    )

    selected_keys = {
        (
            q["source_id"],
            q["question_id"],
        )
        for q in gold
    }

    all_rows = [
        r for r in all_rows
        if r["condition"] in condition_order
        and (
            r["source_id"],
            r["question_id"],
        ) in selected_keys
    ]

    metrics = aggregate_all(
        args.dataset,
        all_rows,
        condition_order,
    )

    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "primary_metric": (
            "llm_judge_accuracy_cat1_4"
            if args.dataset == "locomo"
            else "llm_judge_accuracy"
        ),
        "retention_definition": (
            "condition main_accuracy / full_context main_accuracy"
        ),
        "judge": judge.metadata(),
        "prompt_versions": {
            "locomo": LOCOMO_PROMPT_VERSION,
            "longmemeval": LONGMEMEVAL_PROMPT_VERSION,
        },
        "conditions": metrics,
        "judge_diagnostics": all_judge_diagnostics(
            all_rows,
            condition_order,
        ),
        "comparisons": {
            "stage1_vs_stage2": stage1_stage2_analysis(
                args.dataset,
                all_rows,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
        },
    }

    metrics_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    write_table(
        table_path,
        args.dataset,
        metrics,
        condition_order,
    )

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    for condition in condition_order:
        m = metrics[condition]

        print(
            f"{condition:<15} "
            f"main_acc={m['main_accuracy']!s:<12} "
            f"retention={m['performance_retention']!s}"
        )

        if args.dataset == "locomo":
            print(
                f"{'':<15} "
                f"adv_acc={m['adversarial_accuracy']!s:<12} "
                f"native_main={m['native_main_score_cat1_4']!s}"
            )
        else:
            print(
                f"{'':<15} "
                f"abstention_acc={m['abstention_accuracy']!s}"
            )

    comparison = summary["comparisons"]["stage1_vs_stage2"]
    if comparison is not None:
        tm = comparison["transition_matrix"]
        mc = comparison["mcnemar_exact"]
        bs = comparison["paired_bootstrap"]

        print("-" * 78)
        print("STAGE-1 vs STAGE-2 PAIRED ANALYSIS")
        print(
            f"paired n={comparison['paired_questions']} | "
            f"0->1={tm['0_to_1']} | "
            f"1->0={tm['1_to_0']} | "
            f"net={tm['net_correct_gain']:+d}"
        )
        print(
            "accuracy delta (stage2-stage1): "
            f"{tm['accuracy_delta_stage2_minus_stage1']}"
        )
        print(
            "McNemar exact two-sided p: "
            f"{mc['p_value_exact_two_sided']}"
        )
        print(
            "bootstrap 95% CI, accuracy delta: "
            f"{bs['accuracy_delta']['ci95_percentile']}"
        )
        print(
            "bootstrap 95% CI, retention delta: "
            f"{bs['retention_delta']['ci95_percentile']}"
        )

    print("=" * 78)
    print(f"per question : {perq_path}")
    print(f"metrics      : {metrics_path}")
    print(f"table        : {table_path}")
    print(f"run config   : {config_path}")


if __name__ == "__main__":
    main()
