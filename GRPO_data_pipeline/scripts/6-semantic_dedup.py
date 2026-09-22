#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_data.io_utils import extract_first_json_object, read_jsonl
from stage2_data.llm import add_llm_args, build_client
from stage2_data.prompts import DEDUP_SYSTEM, dedup_user_prompt
from stage2_data.runner import run_resumable
from stage2_data.schema import parse_duplicate_groups


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--errors", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    add_llm_args(p)
    args = p.parse_args()

    client = build_client(args)
    records = list(read_jsonl(args.input))
    if args.limit is not None:
        records = records[:args.limit]

    error_path = args.errors or str(Path(args.output).with_suffix(".errors.jsonl"))

    def process(record):
        units = record.get("units", [])
        response = client.generate(DEDUP_SYSTEM, dedup_user_prompt(units))
        payload = extract_first_json_object(response)
        groups = parse_duplicate_groups(payload)

        unit_map = {u["unit_id"]: u for u in units}
        valid_ids = set(unit_map)
        uf = UnionFind(valid_ids)
        preferred_reps = []
        warnings = []

        for group in groups:
            members = [x for x in group["member_ids"] if x in valid_ids]
            unknown = [x for x in group["member_ids"] if x not in valid_ids]
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

        components = {}
        for uid in valid_ids:
            root = uf.find(uid)
            components.setdefault(root, []).append(uid)

        final_units = []
        for members in components.values():
            member_set = set(members)
            rep = next((r for r in preferred_reps if r in member_set), None)
            if rep is None:
                rep = sorted(members)[0]
            src = unit_map[rep]

            provenance = []
            types_seen = []
            for mid in sorted(members):
                u = unit_map[mid]
                provenance.extend(u.get("provenance", []))
                for t in u.get("types_seen", [u.get("type", "other")]):
                    if t not in types_seen:
                        types_seen.append(t)

            final_units.append({
                "unit_id": rep,
                "claim": src["claim"],
                "type": src.get("type", "other"),
                "types_seen": types_seen,
                "merged_from": sorted(members),
                "provenance": provenance,
            })

        final_units.sort(key=lambda x: x["unit_id"])

        result = {
            "id": record["id"],
            "context_id": record["context_id"],
            "pre_semantic_dedup_count": len(units),
            "final_unit_count": len(final_units),
            "units": final_units,
            "dedup_groups": groups,
            "warnings": warnings,
        }
        if args.keep_raw:
            result["raw_response"] = response
        return result

    workers = args.workers if client.supports_parallel else 1
    run_resumable(
        records, args.output, error_path, process,
        workers=workers, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
