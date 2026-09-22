#!/usr/bin/env python3
"""
Build a one-time Group-2 gold atomic-unit cache for GovReport / MultiNews.

This script does NOT define a new extraction prompt. It reuses the existing
Stage-2 data pipeline's atomic-information extraction and semantic-dedup logic:

    stage2_data.prompts.LOCAL_SYSTEM
    stage2_data.prompts.local_user_prompt
    stage2_data.prompts.DEDUP_SYSTEM
    stage2_data.prompts.dedup_user_prompt
    stage2_data.schema.LOCAL_ATOMIC_JSON_SCHEMA
    stage2_data.schema.parse_local_units_payload
    stage2_data.schema.parse_duplicate_groups
    stage2_data.llm.build_client
    stage2_data.runner.run_resumable

Important evaluation contract:
- ONLY the benchmark GOLD SUMMARY is atomized.
- Compressor outputs are NOT atomized for the primary coverage metric.
- Gold summaries are short, so this script feeds each full gold summary directly
  to the existing local atomic extractor: no chunking and no global-relation pass.
- Exact normalized duplicates are removed deterministically.
- Semantic deduplication reuses the original Stage-2 dedup prompt.
- The resulting cache is generated once and reused for every condition/checkpoint.

Input eval_meta JSONL:
{
  "id": "...",
  "dataset": "gov_report" | "multi_news",
  "gold_summaries": ["..."]
}

Output cache JSONL:
{
  "id": "...",
  "dataset": "...",
  "gold_summary": "...",
  "raw_unit_count": 27,
  "exact_dedup_unit_count": 25,
  "final_unit_count": 23,
  "units": [
    {
      "unit_id": "u00001",
      "claim": "...",
      "merged_from": ["u00001"],
      "provenance": [
        {
          "source": "gold_summary",
          "evidence": ["..."],
          "evidence_exact_match": [true]
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------
# Pre-parse pipeline root so we can import the ORIGINAL extraction package.
# ---------------------------------------------------------------------

def infer_default_pipeline_root() -> str:
    here = Path(__file__).resolve()

    # Intended installed location:
    #   <project>/experience/scripts/eval/prepare_atomic_cache.py
    # so parents[3] is <project>.
    if len(here.parents) >= 4:
        project_root = here.parents[3]

        # Current project layout first.
        candidate = project_root / "GRPO_data_pipeline"
        if (candidate / "stage2_data").is_dir():
            return str(candidate)

        # Backward-compatible fallback for the older package layout.
        candidate = project_root / "quality_stage2_pipeline"
        if (candidate / "stage2_data").is_dir():
            return str(candidate)

        # Return the current expected location so the eventual error message
        # is useful even when neither directory exists.
        return str(project_root / "GRPO_data_pipeline")

    return "GRPO_data_pipeline"


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument(
    "--pipeline-root",
    default=infer_default_pipeline_root(),
)
_pre_args, _ = _pre.parse_known_args()

PIPELINE_ROOT = Path(_pre_args.pipeline_root).resolve()
if not (PIPELINE_ROOT / "stage2_data").is_dir():
    raise FileNotFoundError(
        f"Stage-2 extraction package not found: {PIPELINE_ROOT / 'stage2_data'}\n"
        "Pass --pipeline-root to the directory that directly contains stage2_data/."
    )

sys.path.insert(0, str(PIPELINE_ROOT))

from stage2_data.io_utils import (  # noqa: E402
    ModelOutputError,
    extract_json_value,
    normalize_claim,
    normalize_ws,
    read_jsonl,
)
from stage2_data.llm import add_llm_args, build_client  # noqa: E402
from stage2_data.prompts import (  # noqa: E402
    DEDUP_SYSTEM,
    LOCAL_SYSTEM,
    dedup_user_prompt,
    local_user_prompt,
)
from stage2_data.runner import run_resumable  # noqa: E402
from stage2_data.schema import (  # noqa: E402
    LOCAL_ATOMIC_JSON_SCHEMA,
    parse_duplicate_groups,
    parse_local_units_payload,
)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pipeline-root",
        default=str(PIPELINE_ROOT),
        help=(
            "Existing Stage-2 pipeline root that directly contains stage2_data/. "
            "No extraction prompt is duplicated in this script."
        ),
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["gov_report", "multi_news"],
    )
    p.add_argument(
        "--eval-meta",
        required=True,
        help="experience/processed/group2/<dataset>/eval_meta.jsonl",
    )
    p.add_argument(
        "--output",
        required=True,
        help="One-time cached gold atomic units JSONL.",
    )
    p.add_argument("--errors", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument(
        "--no-semantic-dedup",
        dest="semantic_dedup",
        action="store_false",
        help=(
            "Disable the original Stage-2 semantic duplicate detector. "
            "Formal evaluation should normally leave dedup enabled."
        ),
    )
    p.set_defaults(semantic_dedup=True)

    add_llm_args(p)

    args = p.parse_args()

    # The import path was fixed during pre-parsing; refuse a mismatched second
    # value rather than silently importing one pipeline and reporting another.
    if Path(args.pipeline_root).resolve() != PIPELINE_ROOT:
        p.error(
            "--pipeline-root changed after the extraction package was imported. "
            "Use one consistent --pipeline-root value for the command."
        )

    return args


# ---------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------

def extract_gold_summary(
    row: Dict[str, Any],
    dataset: str,
) -> str:
    row_dataset = str(row.get("dataset", dataset))
    if row_dataset != dataset:
        raise ValueError(
            f"{row.get('id')}: dataset={row_dataset!r}, expected {dataset!r}"
        )

    golds = row.get("gold_summaries")
    if not isinstance(golds, list):
        raise ValueError(
            f"{row.get('id')}: gold_summaries must be a list"
        )

    golds = [
        x.strip()
        for x in golds
        if isinstance(x, str) and x.strip()
    ]

    if len(golds) != 1:
        raise ValueError(
            f"{row.get('id')}: expected exactly one non-empty gold summary, "
            f"found {len(golds)}"
        )

    return golds[0]


# ---------------------------------------------------------------------
# Exact dedup: same normalized-claim rule as original merge stage.
# ---------------------------------------------------------------------

def exact_dedup_units(
    units: List[Dict[str, Any]],
    gold_summary: str,
) -> List[Dict[str, Any]]:
    """Deterministically remove exact normalized duplicate claims.

    The CURRENT local extractor returns:
        {"claim": "...", "evidence": "..."}
    where evidence is a single exact source span string.
    """
    summary_norm = normalize_ws(gold_summary)

    merged: List[Dict[str, Any]] = []
    key_to_index: Dict[str, int] = {}

    for unit in units:
        key = normalize_claim(unit["claim"])
        if not key:
            continue

        evidence = normalize_ws(unit.get("evidence", ""))
        provenance = [{
            "source": "gold_summary",
            "evidence": [evidence] if evidence else [],
            "evidence_exact_match": (
                [evidence in summary_norm] if evidence else []
            ),
        }]

        if key in key_to_index:
            dst = merged[key_to_index[key]]
            dst["provenance"].extend(provenance)
            continue

        key_to_index[key] = len(merged)
        merged.append({
            "claim": unit["claim"],
            "provenance": provenance,
        })

    for idx, unit in enumerate(merged, 1):
        unit["unit_id"] = f"u{idx:05d}"

    return merged


# ---------------------------------------------------------------------
# Semantic dedup: same prompt + same grouping semantics as old Step 6.
# ---------------------------------------------------------------------

class UnionFind:
    def __init__(self, ids):
        self.parent = {x: x for x in ids}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def apply_semantic_dedup(
    units: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
):
    unit_map = {u["unit_id"]: u for u in units}
    valid_ids = set(unit_map)

    uf = UnionFind(valid_ids)
    preferred_reps = []
    warnings = []

    for group in groups:
        members = [
            x for x in group["member_ids"]
            if x in valid_ids
        ]
        unknown = [
            x for x in group["member_ids"]
            if x not in valid_ids
        ]

        if unknown:
            warnings.append(f"unknown_ids:{unknown}")

        if len(members) < 2:
            continue

        base = members[0]
        for other in members[1:]:
            uf.union(base, other)

        rep = group["representative_id"]
        if rep in members:
            preferred_reps.append(rep)

    components = collections.defaultdict(list)
    for uid in sorted(valid_ids):
        components[uf.find(uid)].append(uid)

    final_units = []

    for members in components.values():
        member_set = set(members)

        rep = next(
            (
                r for r in preferred_reps
                if r in member_set
            ),
            None,
        )
        if rep is None:
            rep = sorted(members)[0]

        src = unit_map[rep]

        provenance = []

        for mid in sorted(members):
            unit = unit_map[mid]
            provenance.extend(
                unit.get("provenance", [])
            )

        final_units.append({
            "unit_id": rep,
            "claim": src["claim"],
            "merged_from": sorted(members),
            "provenance": provenance,
        })

    final_units.sort(
        key=lambda x: x["unit_id"]
    )

    return final_units, warnings


def identity_final_units(
    units: List[Dict[str, Any]],
):
    return [
        {
            **unit,
            "merged_from": [unit["unit_id"]],
        }
        for unit in units
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    client = build_client(args)
    records = list(
        read_jsonl(args.eval_meta)
    )

    if args.limit is not None:
        records = records[:args.limit]

    output_path = Path(args.output)
    error_path = (
        args.errors
        or str(
            output_path.with_suffix(
                ".errors.jsonl"
            )
        )
    )

    def process(record):
        sid = str(record["id"])
        gold_summary = extract_gold_summary(
            record,
            args.dataset,
        )

        # Reuse the original local atomic extractor exactly:
        # no chunking; the whole short gold summary is the "source chunk".
        extract_response = client.generate(
            LOCAL_SYSTEM,
            local_user_prompt(gold_summary),
            guided_json=LOCAL_ATOMIC_JSON_SCHEMA,
        )

        try:
            # Current local extraction returns a TOP-LEVEL JSON ARRAY:
            # [{"claim": "...", "evidence": "..."}, ...]
            extract_payload = extract_json_value(
                extract_response
            )
            raw_units = parse_local_units_payload(
                extract_payload
            )
            if not raw_units:
                raise ValueError(
                    "Atomic extractor returned zero units."
                )
        except Exception as e:
            raise ModelOutputError(
                str(e),
                raw_response=extract_response,
            ) from e

        exact_units = exact_dedup_units(
            raw_units,
            gold_summary,
        )

        dedup_response = None
        dedup_groups = []
        dedup_warnings = []

        if args.semantic_dedup and len(exact_units) >= 2:
            # Reuse original Step-6 semantic dedup prompt.
            dedup_response = client.generate(
                DEDUP_SYSTEM,
                dedup_user_prompt(exact_units),
            )

            try:
                dedup_payload = extract_json_value(
                    dedup_response
                )
                if not isinstance(dedup_payload, dict):
                    raise ValueError(
                        "Semantic-dedup output must be a top-level JSON object."
                    )
                dedup_groups = parse_duplicate_groups(
                    dedup_payload
                )
            except Exception as e:
                raise ModelOutputError(
                    str(e),
                    raw_response=dedup_response,
                ) from e

            final_units, dedup_warnings = (
                apply_semantic_dedup(
                    exact_units,
                    dedup_groups,
                )
            )
        else:
            final_units = identity_final_units(
                exact_units
            )

        result = {
            "id": sid,
            "dataset": args.dataset,
            "gold_summary": gold_summary,
            "raw_unit_count": len(raw_units),
            "exact_dedup_unit_count": len(
                exact_units
            ),
            "final_unit_count": len(
                final_units
            ),
            "units": final_units,
            "semantic_dedup_enabled": bool(
                args.semantic_dedup
            ),
            "dedup_groups": dedup_groups,
            "warnings": dedup_warnings,
            "extraction_contract": {
                "source": (
                    "existing stage2_data "
                    "LOCAL_SYSTEM/local_user_prompt + "
                    "LOCAL_ATOMIC_JSON_SCHEMA/parse_local_units_payload"
                ),
                "chunking": False,
                "global_relation_pass": False,
                "semantic_dedup": bool(
                    args.semantic_dedup
                ),
            },
        }

        if args.keep_raw:
            result[
                "raw_extraction_response"
            ] = extract_response
            result[
                "raw_dedup_response"
            ] = dedup_response

        return result

    workers = (
        args.workers
        if client.supports_parallel
        else 1
    )

    run_resumable(
        records,
        str(output_path),
        str(error_path),
        process,
        workers=workers,
        overwrite=args.overwrite,
    )

    print("=" * 78)
    print("GROUP-2 GOLD ATOMIC CACHE")
    print("=" * 78)
    print(f"dataset        : {args.dataset}")
    print(f"eval meta      : {args.eval_meta}")
    print(f"pipeline root  : {PIPELINE_ROOT}")
    print(f"output         : {output_path}")
    print(f"semantic dedup : {args.semantic_dedup}")
    print("chunking       : NO")
    print("global pass    : NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
