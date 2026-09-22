cd ~/LoRA/condense/GRPO_data_pipeline

python - <<'PY'
from statistics import mean, median

from stage2_data.io_utils import read_jsonl

base = "work/train/final"

stage2 = list(read_jsonl(f"{base}/stage2_train.jsonl"))
policy = list(read_jsonl(f"{base}/train_policy.jsonl"))
reward = list(read_jsonl(f"{base}/train_reward.jsonl"))

# --------------------------------------------------
# 1. Row counts
# --------------------------------------------------

assert len(stage2) == len(policy) == len(reward), (
    len(stage2), len(policy), len(reward)
)

print(f"stage2 rows : {len(stage2)}")
print(f"policy rows : {len(policy)}")
print(f"reward rows : {len(reward)}")

# --------------------------------------------------
# 2. IDs
# --------------------------------------------------

stage2_ids = [x["id"] for x in stage2]
policy_ids = [x["id"] for x in policy]
reward_ids = [x["id"] for x in reward]

assert len(stage2_ids) == len(set(stage2_ids)), "duplicate stage2 ids"
assert len(policy_ids) == len(set(policy_ids)), "duplicate policy ids"
assert len(reward_ids) == len(set(reward_ids)), "duplicate reward ids"

assert stage2_ids == policy_ids == reward_ids, "id/order mismatch"

print("IDs        : OK")

# --------------------------------------------------
# 3. Policy leakage check
# --------------------------------------------------

allowed_policy_keys = {
    "id",
    "context",
    "source_tokens",
}

for row in policy:
    assert set(row.keys()) == allowed_policy_keys, (
        row["id"],
        set(row.keys()),
    )

    assert isinstance(row["context"], str)
    assert row["context"].strip()

    assert isinstance(row["source_tokens"], int)
    assert row["source_tokens"] > 0

print("policy     : OK (context only, no reward metadata)")

# --------------------------------------------------
# 4. Reward validation
# --------------------------------------------------

unit_counts = []
probe_counts = []

for row in reward:
    rid = row["id"]

    units = row.get("information_units")
    probes = row.get("probes")

    assert isinstance(units, list) and units, (
        rid,
        "empty information_units",
    )

    assert isinstance(probes, list) and probes, (
        rid,
        "empty probes",
    )

    for i, unit in enumerate(units):
        assert isinstance(unit, dict), (rid, i)
        assert set(unit.keys()) == {"claim", "evidence"}, (
            rid,
            i,
            unit.keys(),
        )

        assert isinstance(unit["claim"], str)
        assert unit["claim"].strip()

        assert isinstance(unit["evidence"], str)
        assert unit["evidence"].strip()

    for i, probe in enumerate(probes):
        assert isinstance(probe, dict), (rid, i)

        assert isinstance(probe.get("question"), str)
        assert probe["question"].strip()

        choices = probe.get("choices")
        assert isinstance(choices, list)
        assert len(choices) > 0
        assert all(
            isinstance(x, str) and x.strip()
            for x in choices
        )

        answer = probe.get("answer_letter")
        assert answer in {"A", "B", "C", "D"}, (
            rid,
            i,
            answer,
        )

    unit_counts.append(len(units))
    probe_counts.append(len(probes))

print("reward     : OK")

# --------------------------------------------------
# 5. Unified file alignment
# --------------------------------------------------

for full, p, r in zip(stage2, policy, reward):
    assert full["id"] == p["id"] == r["id"]

    assert full["context"] == p["context"]
    assert full["source_tokens"] == p["source_tokens"]

    assert full["information_units"] == r["information_units"]
    assert full["probes"] == r["probes"]

print("alignment  : OK")

# --------------------------------------------------
# 6. Summary statistics
# --------------------------------------------------

total_units = sum(unit_counts)
total_probes = sum(probe_counts)

print()
print("=" * 60)
print("FINAL DATASET VALIDATION PASSED")
print("=" * 60)

print(f"articles            : {len(policy)}")
print(f"information units   : {total_units}")
print(
    "units/article      : "
    f"min={min(unit_counts)} "
    f"mean={mean(unit_counts):.2f} "
    f"median={median(unit_counts):.1f} "
    f"max={max(unit_counts)}"
)

print(f"total probes        : {total_probes}")
print(
    "probes/article     : "
    f"min={min(probe_counts)} "
    f"mean={mean(probe_counts):.2f} "
    f"median={median(probe_counts):.1f} "
    f"max={max(probe_counts)}"
)

assert len(policy) == 150, f"expected 150 articles, got {len(policy)}"
assert total_probes == 2523, f"expected 2523 probes, got {total_probes}"

print()
print("Expected QuALITY counts: OK")
print("=" * 60)
PY


# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# import statistics
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# from stage2_data.io_utils import read_jsonl


# def describe(values):
#     values = list(values)
#     if not values:
#         return {"count": 0}
#     values_sorted = sorted(values)

#     def pct(p):
#         idx = int(round((len(values_sorted) - 1) * p))
#         return values_sorted[idx]

#     return {
#         "count": len(values),
#         "min": min(values),
#         "mean": statistics.mean(values),
#         "median": statistics.median(values),
#         "p90": pct(0.90),
#         "max": max(values),
#     }


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--input", required=True)
#     p.add_argument("--report", default=None)
#     args = p.parse_args()

#     rows = list(read_jsonl(args.input))
#     unit_counts = [len(x.get("information_units", [])) for x in rows]
#     probe_counts = [len(x.get("probes", [])) for x in rows]
#     source_tokens = [x["source_tokens"] for x in rows if x.get("source_tokens") is not None]

#     missing_answers = 0
#     duplicate_unit_ids = 0
#     empty_claims = 0
#     exact_evidence_checks = []
#     facts_per_1k = []

#     for row in rows:
#         units = row.get("information_units", [])
#         ids = [u.get("id") for u in units]
#         duplicate_unit_ids += len(ids) - len(set(ids))
#         empty_claims += sum(1 for u in units if not str(u.get("claim", "")).strip())
#         missing_answers += sum(
#             1 for q in row.get("probes", []) if q.get("answer_letter") is None
#         )
#         if row.get("source_tokens"):
#             facts_per_1k.append(len(units) * 1000.0 / row["source_tokens"])

#         for u in units:
#             for prov in u.get("provenance", []):
#                 vals = prov.get("evidence_exact_match")
#                 if isinstance(vals, list):
#                     exact_evidence_checks.extend(bool(x) for x in vals)

#     report = {
#         "articles": len(rows),
#         "source_tokens": describe(source_tokens),
#         "information_units_per_article": describe(unit_counts),
#         "probes_per_article": describe(probe_counts),
#         "facts_per_1k_source_tokens": describe(facts_per_1k),
#         "missing_probe_answers": missing_answers,
#         "duplicate_unit_ids": duplicate_unit_ids,
#         "empty_claims": empty_claims,
#         "local_evidence_exact_match_rate": (
#             sum(exact_evidence_checks) / len(exact_evidence_checks)
#             if exact_evidence_checks else None
#         ),
#     }

#     print(json.dumps(report, indent=2, ensure_ascii=False))
#     if args.report:
#         Path(args.report).parent.mkdir(parents=True, exist_ok=True)
#         Path(args.report).write_text(
#             json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
#         )
#         print(f"report={args.report}")


# if __name__ == "__main__":
#     main()
