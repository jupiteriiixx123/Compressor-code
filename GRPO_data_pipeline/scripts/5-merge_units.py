#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_data.io_utils import normalize_claim, read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--articles", required=True)
    p.add_argument("--local-units", required=True)
    p.add_argument("--global-units", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    raw_by_context = collections.defaultdict(list)

    for row in read_jsonl(args.local_units):
        for unit in row.get("units", []):
            raw_by_context[row["context_id"]].append({
                "claim": unit["claim"],
                "type": unit.get("type", "other"),
                "provenance": [{
                    "source": "local",
                    "chunk_id": row.get("chunk_id"),
                    "evidence": unit.get("evidence", []),
                    "evidence_exact_match": unit.get("evidence_exact_match", []),
                }],
            })

    for row in read_jsonl(args.global_units):
        for unit in row.get("units", []):
            raw_by_context[row["context_id"]].append({
                "claim": unit["claim"],
                "type": unit.get("type", "other"),
                "provenance": [{
                    "source": "global",
                    "chunk_id": None,
                    "evidence": unit.get("evidence", []),
                    "evidence_chunk_matches": unit.get("evidence_chunk_matches", []),
                }],
            })

    output = []
    for article in read_jsonl(args.articles):
        raw = raw_by_context.get(article["id"], [])
        merged = []
        key_to_index = {}

        for unit in raw:
            key = normalize_claim(unit["claim"])
            if not key:
                continue
            if key in key_to_index:
                dst = merged[key_to_index[key]]
                dst["provenance"].extend(unit["provenance"])
                if unit["type"] not in dst["types_seen"]:
                    dst["types_seen"].append(unit["type"])
                continue

            key_to_index[key] = len(merged)
            merged.append({
                "claim": unit["claim"],
                "type": unit["type"],
                "types_seen": [unit["type"]],
                "provenance": list(unit["provenance"]),
            })

        for idx, unit in enumerate(merged, 1):
            unit["unit_id"] = f"u{idx:05d}"

        output.append({
            "id": article["id"],
            "context_id": article["id"],
            "raw_unit_count": len(raw),
            "exact_dedup_unit_count": len(merged),
            "units": merged,
        })

    write_jsonl(args.output, output)
    print(f"articles={len(output)}")
    print(f"raw_units={sum(x['raw_unit_count'] for x in output)}")
    print(f"after_exact_dedup={sum(x['exact_dedup_unit_count'] for x in output)}")
    print(f"written={args.output}")


if __name__ == "__main__":
    main()
