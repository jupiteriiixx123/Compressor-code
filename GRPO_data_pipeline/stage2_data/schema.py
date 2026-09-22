from __future__ import annotations

from typing import Any, Dict, List

from .io_utils import normalize_ws


# This is the schema passed to vLLM guided_json for LOCAL extraction.
# It constrains the model to output exactly:
# [
#   {"claim": "...", "evidence": "..."},
#   ...
# ]
LOCAL_ATOMIC_JSON_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["claim", "evidence"],
        "additionalProperties": False,
    },
}


def parse_local_units_payload(payload: Any) -> List[Dict[str, str]]:
    """Validate local atomic units, dropping invalid individual units."""
    if not isinstance(payload, list):
        raise ValueError("Local extraction JSON must be a top-level list.")

    units: List[Dict[str, str]] = []

    for raw in payload:
        if not isinstance(raw, dict):
            continue

        claim = normalize_ws(raw.get("claim", ""))
        evidence = normalize_ws(raw.get("evidence", ""))

        # Drop only the invalid unit instead of failing the whole chunk.
        if not claim or not evidence:
            continue

        units.append({
            "claim": claim,
            "evidence": evidence,
        })

    # If the entire chunk contains no usable units, still treat it as failed.
    if not units:
        raise ValueError("Local extraction produced no valid units.")

    return units


# ---- Legacy/global schema support -----------------------------------------
# Global extraction is intentionally left compatible with the existing Step-4
# script. We are not redesigning that stage while debugging local extraction.
ALLOWED_TYPES = {
    "entity_attribute", "relation", "event", "state", "state_change",
    "temporal", "spatial", "numeric", "causal", "intention",
    "negation", "comparison", "other",
}


def _normalize_type(value: Any) -> str:
    value = normalize_ws(value or "other").lower().replace(" ", "_").replace("-", "_")
    return value if value in ALLOWED_TYPES else "other"


def normalize_unit(raw: Dict[str, Any]) -> Dict[str, Any] | None:
    claim = normalize_ws(raw.get("claim", ""))
    if not claim:
        return None
    evidence_raw = raw.get("evidence", [])
    if isinstance(evidence_raw, str):
        evidence = [normalize_ws(evidence_raw)] if normalize_ws(evidence_raw) else []
    elif isinstance(evidence_raw, list):
        evidence = [normalize_ws(x) for x in evidence_raw if normalize_ws(x)]
    else:
        evidence = []
    return {"claim": claim, "type": _normalize_type(raw.get("type")), "evidence": evidence}


def parse_units_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Backward-compatible parser used by the existing global extraction script."""
    raw_units = payload.get("units")
    if not isinstance(raw_units, list):
        raise ValueError("Model JSON must contain a list field named 'units'.")
    units = []
    for raw in raw_units:
        if not isinstance(raw, dict):
            continue
        unit = normalize_unit(raw)
        if unit is not None:
            units.append(unit)
    return units


def parse_duplicate_groups(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_groups = payload.get("duplicate_groups", [])
    if not isinstance(raw_groups, list):
        raise ValueError("'duplicate_groups' must be a list.")
    out = []
    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        members = group.get("member_ids", [])
        if not isinstance(members, list):
            continue
        members = [str(x) for x in members if str(x).strip()]
        rep = str(group.get("representative_id", "")).strip()
        if len(members) >= 2:
            out.append({"representative_id": rep, "member_ids": members})
    return out
