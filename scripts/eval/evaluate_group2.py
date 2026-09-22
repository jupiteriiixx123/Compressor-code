#!/usr/bin/env python3
from __future__ import annotations
"""
Group-2 evaluator for GovReport / MultiNews.

This file intentionally reuses:
  1) the project's EXISTING atomic-information extraction pipeline;
  2) the EXISTING eval/judge.py BinaryJudge.

It does NOT contain decomposition logic and does NOT define another Judge.

Expected inputs
---------------
- compressor_input.jsonl:
    {"id": ..., "context": ..., "source_tokens": ...}

- eval_meta.jsonl:
    {"id": ..., "gold_summaries": [...]}

- gold atomic units produced by the existing extraction pipeline.
  Two article-level schemas are accepted:

  A) Stage-2 reward style
    {
      "id": "...",
      "information_units": [
        {"id": "u00001", "claim": "...", "type": "...", "provenance": [...]}
      ]
    }

  B) Generic units style
    {
      "id": "...",
      "units": [
        {"id": "u00001", "claim": "..."}
      ]
    }

  Unit text may be stored under "claim" or "text".
  Unit id may be stored under "id" or "unit_id".

Primary metrics
---------------
Per sample:
    coverage_i = supported_gold_units / total_gold_units

Dataset:
    Coverage_macro = mean_i(coverage_i)

    Full_CR_corpus =
        sum_i tokens(exact_flattened_compression_i)
        / sum_i tokens(source_context_i)

    Backbone_CR_corpus =
        sum_i tokens(summary_i)
        / sum_i tokens(source_context_i)

    Information_Density =
        Coverage_macro / Full_CR_corpus

The primary ratio always uses the FULL eager-flattened representation because
coverage is judged against that same representation.

Full Context is evaluated with the same support Judge.  Its compression ratio
is exactly 1, so Density_full == Coverage_full.

No Reader generation is used.
"""

# ======================================================


"""Minimal Group-2 evaluator.

Runs exactly the conditions passed with --compression.

Primary metrics:
  Coverage_macro = mean_i(supported_gold_units_i / gold_units_i)
  Full_CR        = sum(tokens(flattened compression)) / sum(tokens(source))
  Backbone_CR    = sum(tokens(summary backbone)) / sum(tokens(source))
  Density        = Coverage_macro / Full_CR
  ROUGE-L        = LongBench-native summary overlap metric

Outputs:
  per_unit.jsonl, per_sample.jsonl, metrics.json, table.csv, run_config.json
"""


import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from judge import BinaryJudge


SUPPORT_SYSTEM_PROMPT = """You are a strict information-recoverability evaluator.

Given a MEMORY and one GOLD ATOMIC INFORMATION UNIT, decide whether the entire
unit is recoverable from the memory.

Return label 1 only if the memory explicitly states the unit or directly
entails it through a clear paraphrase or unambiguous coreference resolution.

Return label 0 if the information is absent, contradicted, only partially
supported, materially less specific, or would require outside knowledge or
substantial inference.

Do not reward topical similarity. All essential entities, relations,
quantities, dates, negation, modality, and attribution in the gold unit must
be supported.

Do not produce explanations. The scoring code evaluates only the fixed
candidate labels 0 and 1."""
SUPPORT_PROMPT_VERSION = "group2_atomic_support_v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["gov_report", "multi_news"])
    p.add_argument("--compressor-input", required=True)
    p.add_argument("--eval-meta", required=True)
    p.add_argument("--gold-units", required=True)
    p.add_argument(
        "--compression", action="append", required=True, metavar="CONDITION=PATH",
        help="Repeat as needed, e.g. --compression stage1=... --compression stage2=...",
    )
    p.add_argument("--output-dir", required=True)

    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--judge-device-map", default="auto")
    p.add_argument(
        "--judge-attn-implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        default="sdpa",
    )
    p.add_argument("--judge-batch-size", type=int, default=1)
    p.add_argument("--ratio-tokenizer", required=True)

    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.set_defaults(trust_remote_code=True)

    args = p.parse_args()
    if args.judge_batch_size < 1:
        p.error("--judge-batch-size must be >= 1")
    if args.max_samples is not None and args.max_samples < 1:
        p.error("--max-samples must be >= 1")
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


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_by_id(rows: Sequence[Dict[str, Any]], name: str):
    out = {}
    for row in rows:
        if "id" not in row:
            raise ValueError(f"{name}: row missing id")
        sid = str(row["id"])
        if sid in out:
            raise ValueError(f"{name}: duplicate id {sid}")
        out[sid] = row
    return out


def parse_compressions(values: Sequence[str]) -> Dict[str, Path]:
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --compression {value!r}; expected CONDITION=PATH")
        condition, path = (x.strip() for x in value.split("=", 1))
        if not condition or not path:
            raise ValueError(f"Invalid --compression {value!r}; expected CONDITION=PATH")
        if condition in out:
            raise ValueError(f"Duplicate condition: {condition}")
        out[condition] = Path(path)
    return out


def load_atomic_units(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Current cache uses row['units']; older aliases are accepted for compatibility."""
    rows = index_by_id(read_jsonl(path), "gold_units")
    out = {}

    for sid, row in rows.items():
        raw_units = next(
            (row[k] for k in ("units", "information_units", "final_units") if k in row),
            None,
        )
        if not isinstance(raw_units, list) or not raw_units:
            raise ValueError(f"{sid}: no non-empty atomic unit list")

        units, seen = [], set()
        for i, item in enumerate(raw_units, 1):
            if not isinstance(item, dict):
                raise ValueError(f"{sid}: unit {i} is not an object")

            claim = item.get("claim", item.get("text"))
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError(f"{sid}: unit {i} has no claim/text")

            uid = str(item.get("unit_id", item.get("id", f"u{i:05d}")))
            if uid in seen:
                raise ValueError(f"{sid}: duplicate unit id {uid}")
            seen.add(uid)
            units.append({"unit_id": uid, "claim": claim.strip()})

        out[sid] = units
    return out


def single_gold_summary(meta: Dict[str, Any]) -> str:
    golds = meta.get("gold_summaries")
    if not isinstance(golds, list):
        raise ValueError(f"{meta.get('id')}: gold_summaries must be a list")

    golds = [x.strip() for x in golds if isinstance(x, str) and x.strip()]
    if len(golds) != 1:
        raise ValueError(
            f"{meta.get('id')}: expected exactly one non-empty gold summary, "
            f"found {len(golds)}"
        )
    return golds[0]


def make_rouge():
    """
    LongBench's native summarization metric uses the `rouge` package and
    reports ROUGE-L F1. We also keep ROUGE-1/2 F1 as diagnostics.
    """
    try:
        from rouge import Rouge
    except ImportError as e:
        raise RuntimeError(
            "ROUGE evaluation requires the `rouge` Python package, "
            "matching the original LongBench metric implementation."
        ) from e
    return Rouge()


def rouge_scores(rouge, pred: str, gold: str) -> Dict[str, float]:
    try:
        scores = rouge.get_scores([pred], [gold], avg=True)
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    return {
        "rouge1": float(scores["rouge-1"]["f"]),
        "rouge2": float(scores["rouge-2"]["f"]),
        "rougeL": float(scores["rouge-l"]["f"]),
    }


def flatten_compression(row: Dict[str, Any]) -> Tuple[str, str | None]:
    """Exactly match the Reader-view flattening used elsewhere in the project."""
    obj = row.get("output")
    if isinstance(obj, dict):
        summary, link = obj.get("summary"), obj.get("link")
        if isinstance(summary, str) and isinstance(link, dict):
            parts = [f"Summary:\n{summary}"]
            details = [
                f"[{k}] {v}" for k, v in link.items()
                if isinstance(k, str) and isinstance(v, str)
            ]
            if details:
                parts.append("Expandable information:\n" + "\n".join(details))
            return "\n\n".join(parts), summary

    raw = row.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        return raw.strip(), None
    raise ValueError(f"{row.get('id')}: no usable output/raw_output")


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def align_ids(source_rows, meta_rows, atomic_units, compression_rows, allow_missing, max_samples):
    source_ids = list(source_rows)
    if max_samples is not None:
        source_ids = source_ids[:max_samples]

    missing_meta = [sid for sid in source_ids if sid not in meta_rows]
    missing_units = [sid for sid in source_ids if sid not in atomic_units]
    missing_comp = {
        c: [sid for sid in source_ids if sid not in rows]
        for c, rows in compression_rows.items()
    }

    if not allow_missing:
        if missing_meta:
            raise ValueError(
                f"Missing eval_meta for {len(missing_meta)} samples; first={missing_meta[:3]}"
            )
        if missing_units:
            raise ValueError(
                f"Missing gold units for {len(missing_units)} samples; first={missing_units[:3]}"
            )
        for c, missing in missing_comp.items():
            if missing:
                raise ValueError(
                    f"{c}: missing compression for {len(missing)} samples; first={missing[:3]}"
                )
        return source_ids

    selected = [
        sid for sid in source_ids
        if sid in meta_rows
        and sid in atomic_units
        and all(sid in rows for rows in compression_rows.values())
    ]
    if not selected:
        raise ValueError("--allow-missing produced an empty common intersection")

    excluded = len(source_ids) - len(selected)
    if excluded:
        print(
            f"[allow-missing] common intersection: "
            f"{len(selected)}/{len(source_ids)} samples; excluded={excluded}"
        )
    return selected


def build_support_prompt(memory: str, claim: str):
    user = f"""MEMORY:
{memory}

GOLD ATOMIC INFORMATION UNIT:
{claim}

Is the entire gold atomic information unit recoverable from the memory?
Return the correctness label."""
    return SUPPORT_SYSTEM_PROMPT, user


def load_completed(path: Path):
    completed = set()
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        key = (str(row["condition"]), str(row["source_id"]), str(row["unit_id"]))
        if key in completed:
            raise RuntimeError(f"Duplicate existing judgment {key}")
        completed.add(key)
    return completed


def aggregate_per_sample(
    sample_ids, conditions, atomic_units, accounting, per_unit_rows,
    meta_rows, memories, compression_rows,
):
    labels = {
        (str(r["condition"]), str(r["source_id"]), str(r["unit_id"])): int(r["judge_label"])
        for r in per_unit_rows
    }
    rows = []

    for condition in conditions:
        for sid in sample_ids:
            units = atomic_units[sid]
            unit_labels = []
            for unit in units:
                key = (condition, sid, unit["unit_id"])
                if key not in labels:
                    raise RuntimeError(f"Missing atomic judgment: {key}")
                unit_labels.append(labels[key])

            supported = sum(unit_labels)
            gold = single_gold_summary(meta_rows[sid])
            flat = memories[(condition, sid)]
            _, backbone = flatten_compression(compression_rows[condition][sid])

            rouge = make_rouge()
            rouge_flat = rouge_scores(rouge, flat, gold)
            rouge_backbone = (
                rouge_scores(rouge, backbone, gold)
                if backbone is not None
                else None
            )

            rows.append({
                "condition": condition,
                "source_id": sid,
                "num_gold_units": len(units),
                "supported_units": int(supported),
                "coverage": supported / len(units),
                **accounting[(condition, sid)],
                "rouge_flattened": rouge_flat,
                "rouge_backbone": rouge_backbone,
            })
    return rows


def aggregate_metrics(conditions, sample_rows, per_unit_rows):
    metrics = {}
    for condition in conditions:
        rows = [r for r in sample_rows if r["condition"] == condition]
        unit_rows = [r for r in per_unit_rows if r["condition"] == condition]

        coverage_macro = sum(r["coverage"] for r in rows) / len(rows)
        supported = sum(r["supported_units"] for r in rows)
        total_units = sum(r["num_gold_units"] for r in rows)

        source_tokens = sum(r["source_tokens"] for r in rows)
        full_tokens = sum(r["full_tokens"] for r in rows)
        full_cr = full_tokens / source_tokens

        have_backbone = all(r["backbone_tokens"] is not None for r in rows)
        backbone_tokens = (
            sum(r["backbone_tokens"] for r in rows) if have_backbone else None
        )
        backbone_cr = (
            backbone_tokens / source_tokens if backbone_tokens is not None else None
        )

        flat_rouge = [r["rouge_flattened"] for r in rows]
        backbone_rouge = [
            r["rouge_backbone"] for r in rows
            if r["rouge_backbone"] is not None
        ]

        metrics[condition] = {
            "n_samples": len(rows),
            "n_unit_judgments": len(unit_rows),
            "coverage_macro": coverage_macro,
            "coverage_micro": supported / total_units,
            "full_compression_ratio_corpus": full_cr,
            "backbone_compression_ratio_corpus": backbone_cr,
            "information_density_primary": coverage_macro / full_cr,

            # LongBench-native metric is ROUGE-L F1.
            "rougeL_backbone_macro": (
                sum(x["rougeL"] for x in backbone_rouge) / len(backbone_rouge)
                if backbone_rouge else None
            ),
            "rougeL_flattened_macro": (
                sum(x["rougeL"] for x in flat_rouge) / len(flat_rouge)
            ),

            # Extra diagnostics; not the official LongBench headline metric.
            "rouge1_backbone_macro": (
                sum(x["rouge1"] for x in backbone_rouge) / len(backbone_rouge)
                if backbone_rouge else None
            ),
            "rouge2_backbone_macro": (
                sum(x["rouge2"] for x in backbone_rouge) / len(backbone_rouge)
                if backbone_rouge else None
            ),
            "rouge1_flattened_macro": (
                sum(x["rouge1"] for x in flat_rouge) / len(flat_rouge)
            ),
            "rouge2_flattened_macro": (
                sum(x["rouge2"] for x in flat_rouge) / len(flat_rouge)
            ),
        }
    return metrics


def write_table(path: Path, conditions, metrics):
    fields = [
        "condition",
        "n_samples",
        "n_unit_judgments",
        "coverage_macro",
        "coverage_micro",
        "full_compression_ratio_corpus",
        "backbone_compression_ratio_corpus",
        "information_density_primary",
        "rougeL_backbone_macro",
        "rougeL_flattened_macro",
        "rouge1_backbone_macro",
        "rouge2_backbone_macro",
        "rouge1_flattened_macro",
        "rouge2_flattened_macro",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for condition in conditions:
            row = {"condition": condition, **metrics[condition]}
            writer.writerow({k: row.get(k) for k in fields})


def main():
    args = parse_args()

    source_rows = index_by_id(
        read_jsonl(Path(args.compressor_input)), "compressor_input"
    )
    meta_rows = index_by_id(
        read_jsonl(Path(args.eval_meta)), "eval_meta"
    )
    atomic_units = load_atomic_units(Path(args.gold_units))

    compression_paths = parse_compressions(args.compression)
    conditions = list(compression_paths)
    compression_rows = {
        c: index_by_id(read_jsonl(path), f"compression:{c}")
        for c, path in compression_paths.items()
    }

    sample_ids = align_ids(
        source_rows, meta_rows, atomic_units, compression_rows,
        args.allow_missing, args.max_samples
    )

    for sid in sample_ids:
        dataset = str(meta_rows[sid].get("dataset", args.dataset))
        if dataset != args.dataset:
            raise ValueError(
                f"{sid}: eval_meta dataset={dataset!r}, expected {args.dataset!r}"
            )
        single_gold_summary(meta_rows[sid])

    tokenizer = AutoTokenizer.from_pretrained(
        args.ratio_tokenizer, trust_remote_code=args.trust_remote_code
    )

    memories, accounting = {}, {}
    token_mismatches = 0

    for sid in sample_ids:
        source = source_rows[sid].get("context")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{sid}: source context is empty")

        source_tokens = token_count(tokenizer, source)
        stored_tokens = source_rows[sid].get("source_tokens")
        if stored_tokens is not None and int(stored_tokens) != source_tokens:
            token_mismatches += 1

        for condition in conditions:
            flat, backbone = flatten_compression(compression_rows[condition][sid])
            full_tokens = token_count(tokenizer, flat)
            backbone_tokens = token_count(tokenizer, backbone) if backbone is not None else None

            memories[(condition, sid)] = flat
            accounting[(condition, sid)] = {
                "source_tokens": source_tokens,
                "full_tokens": full_tokens,
                "backbone_tokens": backbone_tokens,
                "full_compression_ratio": full_tokens / source_tokens,
                "backbone_compression_ratio": (
                    backbone_tokens / source_tokens if backbone_tokens is not None else None
                ),
            }

    total_units = sum(len(atomic_units[sid]) for sid in sample_ids)
    print("=" * 78)
    print("GROUP-2 INFORMATION EVALUATION")
    print("=" * 78)
    print(f"dataset                 : {args.dataset}")
    print(f"samples                 : {len(sample_ids)}")
    print(f"gold atomic units       : {total_units}")
    print(f"conditions              : {', '.join(conditions)}")
    print(f"planned unit judgments  : {total_units * len(conditions)}")
    print(f"ratio tokenizer         : {args.ratio_tokenizer}")
    print(f"source-token mismatches : {token_mismatches}")
    print("=" * 78)

    if args.validate_only:
        print("Validation passed. Judge was not loaded.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_unit": output_dir / "per_unit.jsonl",
        "per_sample": output_dir / "per_sample.jsonl",
        "metrics": output_dir / "metrics.json",
        "table": output_dir / "table.csv",
        "config": output_dir / "run_config.json",
    }

    if args.overwrite:
        for path in paths.values():
            if path.exists():
                path.unlink()

    config = {
        "dataset": args.dataset,
        "compressor_input": str(Path(args.compressor_input).resolve()),
        "eval_meta": str(Path(args.eval_meta).resolve()),
        "gold_units": str(Path(args.gold_units).resolve()),
        "compressions": {k: str(v.resolve()) for k, v in compression_paths.items()},
        "ratio_tokenizer": args.ratio_tokenizer,
        "judge_model": args.judge_model,
        "judge_dtype": args.judge_dtype,
        "judge_device_map": args.judge_device_map,
        "judge_attn_implementation": args.judge_attn_implementation,
        "support_prompt_version": SUPPORT_PROMPT_VERSION,
        "allow_missing": args.allow_missing,
        "max_samples": args.max_samples,
    }

    if paths["config"].exists():
        old = json.loads(paths["config"].read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError(
                "Existing output directory has a different run config. "
                "Use another --output-dir or pass --overwrite."
            )
    else:
        paths["config"].write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    completed = load_completed(paths["per_unit"])
    jobs = []

    for condition in conditions:
        for sid in sample_ids:
            memory = memories[(condition, sid)]
            for unit in atomic_units[sid]:
                key = (condition, sid, unit["unit_id"])
                if key in completed:
                    continue
                system, user = build_support_prompt(memory, unit["claim"])
                jobs.append({
                    "condition": condition,
                    "source_id": sid,
                    "unit": unit,
                    "system_prompt": system,
                    "user_prompt": user,
                })

    print(f"already judged          : {len(completed)}")
    print(f"pending judgments       : {len(jobs)}")

    if jobs:
        judge = BinaryJudge(
            args.judge_model,
            dtype=args.judge_dtype,
            device_map=args.judge_device_map,
            attn_implementation=args.judge_attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )

        mode = "a" if paths["per_unit"].exists() else "w"
        with paths["per_unit"].open(mode, encoding="utf-8") as fout:
            for start in tqdm(
                range(0, len(jobs), args.judge_batch_size),
                desc="Atomic support judging",
            ):
                batch = jobs[start : start + args.judge_batch_size]
                results = judge.score_batch([
                    {"system_prompt": j["system_prompt"], "user_prompt": j["user_prompt"]}
                    for j in batch
                ])
                if len(results) != len(batch):
                    raise RuntimeError("Judge batch/result length mismatch")

                for job, result in zip(batch, results):
                    unit = job["unit"]
                    row = {
                        "dataset": args.dataset,
                        "condition": job["condition"],
                        "source_id": job["source_id"],
                        "unit_id": unit["unit_id"],
                        "atomic_unit": unit["claim"],
                        "judge_label": result.label,
                        "judge_p0": result.p0,
                        "judge_p1": result.p1,
                        "judge_logp0": result.logp0,
                        "judge_logp1": result.logp1,
                        "judge_candidate_mass": result.candidate_mass,
                        "judge_scoring_mode": result.scoring_mode,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()

    selected_ids = set(sample_ids)
    per_unit_rows = [
        r for r in read_jsonl(paths["per_unit"])
        if r["condition"] in conditions and r["source_id"] in selected_ids
    ]

    per_sample_rows = aggregate_per_sample(
        sample_ids, conditions, atomic_units, accounting, per_unit_rows,
        meta_rows, memories, compression_rows
    )
    write_jsonl(paths["per_sample"], per_sample_rows)

    metrics = aggregate_metrics(conditions, per_sample_rows, per_unit_rows)
    paths["metrics"].write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "primary_metrics": {
                    "coverage": "mean sample coverage",
                    "full_compression_ratio": (
                        "corpus flattened-compression tokens / corpus source tokens"
                    ),
                    "information_density": (
                        "coverage_macro / full_compression_ratio_corpus"
                    ),
                    "longbench_native": (
                        "ROUGE-L F1 between benchmark gold summary and compression"
                    ),
                },
                "conditions": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    write_table(paths["table"], conditions, metrics)

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    for condition in conditions:
        m = metrics[condition]
        print(
            f"{condition:<15} "
            f"coverage={m['coverage_macro']:.6f}  "
            f"full_CR={m['full_compression_ratio_corpus']:.6f}  "
            f"density={m['information_density_primary']:.6f}  "
            f"ROUGE-L(backbone)={m['rougeL_backbone_macro']!s}"
        )
        if m["backbone_compression_ratio_corpus"] is not None:
            print(
                f"{'':<15} "
                f"backbone_CR={m['backbone_compression_ratio_corpus']:.6f}"
            )
    print("=" * 78)
    print(f"per unit   : {paths['per_unit']}")
    print(f"per sample : {paths['per_sample']}")
    print(f"metrics    : {paths['metrics']}")
    print(f"table      : {paths['table']}")
    print(f"run config : {paths['config']}")


if __name__ == "__main__":
    main()
