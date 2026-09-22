#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple


# Allow running directly from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_data.io_utils import read_jsonl, write_jsonl, normalize_ws
from stage2_data.schema import parse_local_units_payload


def exact_claim_key(claim: str) -> str:
    """
    Conservative deterministic dedup key.

    Only:
      - normalize whitespace
      - ignore case

    No semantic dedup.
    No stemming.
    No punctuation stripping.
    """
    return normalize_ws(claim).casefold()


def chunk_sort_key(chunk_id: str) -> Tuple[int, str]:
    """
    Sort chunk_0000, chunk_0001, ... in document order.

    Falls back safely if the name is unexpected.
    """
    m = re.search(r"chunk_(\d+)$", chunk_id)
    if m:
        return int(m.group(1)), chunk_id
    return 10**9, chunk_id


def exact_dedup_units(
    units: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], int]:
    """
    Article-level exact claim dedup.

    Keep the first occurrence in document/chunk order.
    """
    seen = set()
    output: List[Dict[str, str]] = []
    removed = 0

    for unit in units:
        claim = normalize_ws(unit["claim"])
        evidence = normalize_ws(unit["evidence"])

        key = exact_claim_key(claim)

        if key in seen:
            removed += 1
            continue

        seen.add(key)

        output.append({
            "claim": claim,
            "evidence": evidence,
        })

    return output, removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build final Stage-2 GRPO dataset from prepared QuALITY "
            "articles and final GLM local atomic units."
        )
    )

    parser.add_argument(
        "--articles",
        required=True,
        help="Prepared articles.jsonl from Step 1.",
    )

    parser.add_argument(
        "--local-units",
        required=True,
        help="Final successful GLM local_units JSONL.",
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "dev", "test"],
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load articles
    # ------------------------------------------------------------------

    articles = list(read_jsonl(args.articles))

    articles_by_id: Dict[str, Dict[str, Any]] = {}

    for article in articles:
        article_id = str(article.get("id", "")).strip()

        if not article_id:
            raise ValueError("Article with missing id.")

        if article_id in articles_by_id:
            raise ValueError(f"Duplicate article id: {article_id}")

        articles_by_id[article_id] = article

    # ------------------------------------------------------------------
    # 2. Load final local-unit chunks
    # ------------------------------------------------------------------

    chunks_by_article: Dict[
        str,
        List[Tuple[str, List[Dict[str, str]]]]
    ] = defaultdict(list)

    seen_local_ids = set()

    local_chunk_count = 0
    raw_unit_count = 0

    for row in read_jsonl(args.local_units):
        row_id = str(row.get("id", "")).strip()

        if not row_id:
            raise ValueError("Local-unit row with missing id.")

        if row_id in seen_local_ids:
            raise ValueError(
                f"Duplicate local-unit row id: {row_id}"
            )

        seen_local_ids.add(row_id)

        # local_units_glm.jsonl should contain successes only.
        if "error" in row:
            raise ValueError(
                f"Error record found in final local-units file: {row_id}"
            )

        context_id = str(row.get("context_id", "")).strip()
        chunk_id = str(row.get("chunk_id", "")).strip()

        if not context_id:
            raise ValueError(
                f"{row_id}: missing context_id."
            )

        if not chunk_id:
            raise ValueError(
                f"{row_id}: missing chunk_id."
            )

        if context_id not in articles_by_id:
            raise ValueError(
                f"{row_id}: unknown context_id {context_id}"
            )

        # Reuse the same local-unit parser used by extraction.
        units = parse_local_units_payload(row.get("units"))

        chunks_by_article[context_id].append(
            (chunk_id, units)
        )

        local_chunk_count += 1
        raw_unit_count += len(units)

    # ------------------------------------------------------------------
    # 3. Aggregate chunk units -> article information_units
    # ------------------------------------------------------------------

    information_units_by_article: Dict[
        str,
        List[Dict[str, str]]
    ] = {}

    exact_duplicates_removed = 0

    for article_id in articles_by_id:
        chunk_rows = chunks_by_article.get(article_id, [])

        if not chunk_rows:
            raise ValueError(
                f"No local units found for article: {article_id}"
            )

        # Important:
        # extraction may have been written in worker-completion order,
        # so explicitly restore chunk/document order.
        chunk_rows.sort(
            key=lambda x: chunk_sort_key(x[0])
        )

        article_units: List[Dict[str, str]] = []

        for _, units in chunk_rows:
            article_units.extend(units)

        article_units, removed = exact_dedup_units(
            article_units
        )

        exact_duplicates_removed += removed

        if not article_units:
            raise ValueError(
                f"{article_id}: no information units after aggregation."
            )

        information_units_by_article[article_id] = article_units

    # ------------------------------------------------------------------
    # 4. Build three final datasets
    # ------------------------------------------------------------------

    unified_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []
    reward_rows: List[Dict[str, Any]] = []

    unit_counts: List[int] = []
    probe_counts: List[int] = []

    for article in articles:
        article_id = article["id"]

        context = article.get("context")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(
                f"{article_id}: empty context."
            )

        if "source_tokens" not in article:
            raise ValueError(
                f"{article_id}: missing source_tokens."
            )

        probes = article.get("probes", [])

        if not isinstance(probes, list) or not probes:
            raise ValueError(
                f"{article_id}: probes must be a non-empty list."
            )

        # Train/dev must have gold answers.
        if args.split in {"train", "dev"}:
            bad = [
                q.get("id", f"probe_{i}")
                for i, q in enumerate(probes)
                if q.get("answer_letter") is None
            ]

            if bad:
                raise ValueError(
                    f"{article_id} has probes without gold answers: "
                    f"{bad[:5]}"
                )

        information_units = information_units_by_article[
            article_id
        ]

        # ---------------------------
        # Full audit version
        # ---------------------------

        unified_rows.append({
            "id": article_id,
            "dataset": article.get("dataset", "quality"),
            "split": args.split,
            "article_id": article.get("article_id"),
            "context": context,
            "source_tokens": article["source_tokens"],
            "information_units": information_units,
            "probes": probes,
            "metadata": article.get("metadata", {}),
        })

        # ---------------------------
        # Compressor-visible policy data
        # ---------------------------

        policy_rows.append({
            "id": article_id,
            "context": context,
            "source_tokens": article["source_tokens"],
        })

        # ---------------------------
        # Hidden reward metadata
        # ---------------------------

        reward_rows.append({
            "id": article_id,
            "information_units": information_units,
            "probes": probes,
        })

        unit_counts.append(len(information_units))
        probe_counts.append(len(probes))

    # ------------------------------------------------------------------
    # 5. Final consistency checks
    # ------------------------------------------------------------------

    if not (
        len(unified_rows)
        == len(policy_rows)
        == len(reward_rows)
        == len(articles)
    ):
        raise RuntimeError(
            "Dataset row counts do not match."
        )

    unified_ids = [x["id"] for x in unified_rows]
    policy_ids = [x["id"] for x in policy_rows]
    reward_ids = [x["id"] for x in reward_rows]

    if unified_ids != policy_ids:
        raise RuntimeError(
            "Unified and policy IDs/order differ."
        )

    if unified_ids != reward_ids:
        raise RuntimeError(
            "Unified and reward IDs/order differ."
        )

    if len(set(unified_ids)) != len(unified_ids):
        raise RuntimeError(
            "Duplicate article IDs in final dataset."
        )

    # Explicit leakage check.
    allowed_policy_keys = {
        "id",
        "context",
        "source_tokens",
    }

    for row in policy_rows:
        if set(row.keys()) != allowed_policy_keys:
            raise RuntimeError(
                f"{row['id']}: unexpected policy fields: "
                f"{set(row.keys()) - allowed_policy_keys}"
            )

    # ------------------------------------------------------------------
    # 6. Write final files
    # ------------------------------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unified_path = (
        output_dir / f"stage2_{args.split}.jsonl"
    )
    policy_path = (
        output_dir / f"{args.split}_policy.jsonl"
    )
    reward_path = (
        output_dir / f"{args.split}_reward.jsonl"
    )

    write_jsonl(unified_path, unified_rows)
    write_jsonl(policy_path, policy_rows)
    write_jsonl(reward_path, reward_rows)

    # ------------------------------------------------------------------
    # 7. Statistics
    # ------------------------------------------------------------------

    final_unit_count = sum(unit_counts)
    total_probes = sum(probe_counts)

    print("=" * 72)
    print("Stage-2 final dataset built successfully")
    print("=" * 72)

    print(f"articles:                  {len(articles)}")
    print(f"local chunks:              {local_chunk_count}")
    print(f"raw local units:           {raw_unit_count}")
    print(
        f"exact duplicates removed:  "
        f"{exact_duplicates_removed}"
    )
    print(
        f"final information units:   "
        f"{final_unit_count}"
    )

    print(
        "units/article:             "
        f"min={min(unit_counts)} "
        f"mean={sum(unit_counts) / len(unit_counts):.2f} "
        f"median={median(unit_counts):.1f} "
        f"max={max(unit_counts)}"
    )

    print(f"total probes:              {total_probes}")

    print(
        "probes/article:            "
        f"min={min(probe_counts)} "
        f"mean={sum(probe_counts) / len(probe_counts):.2f} "
        f"median={median(probe_counts):.1f} "
        f"max={max(probe_counts)}"
    )

    print()
    print(f"unified: {unified_path}")
    print(f"policy:  {policy_path}")
    print(f"reward:  {reward_path}")

    print("=" * 72)


if __name__ == "__main__":
    main()