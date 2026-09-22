#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_data.io_utils import (
    ModelOutputError,
    extract_json_value,
    normalize_ws,
    read_jsonl,
)
from stage2_data.llm import add_llm_args, build_client
from stage2_data.prompts import LOCAL_SYSTEM, local_user_prompt
from stage2_data.runner import run_resumable
from stage2_data.schema import LOCAL_ATOMIC_JSON_SCHEMA, parse_local_units_payload


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
        response = client.generate(
            LOCAL_SYSTEM,
            local_user_prompt(record["text"]),
            guided_json=LOCAL_ATOMIC_JSON_SCHEMA,
        )

        try:
            payload = extract_json_value(response)
            units = parse_local_units_payload(payload)
        except Exception as e:
            raise ModelOutputError(str(e), raw_response=response) from e

        # Evidence remains exactly the simple string produced by the model.
        # We only count mismatches for auditing; no extra field is inserted into
        # each unit, keeping the requested {claim, evidence} structure minimal.
        chunk_norm = normalize_ws(record["text"])
        mismatch_count = sum(
            1 for unit in units
            if normalize_ws(unit["evidence"]) not in chunk_norm
        )

        result = {
            "id": record["id"],
            "context_id": record["context_id"],
            "chunk_id": record["chunk_id"],
            "units": units,
            "evidence_mismatch_count": mismatch_count,
        }
        if args.keep_raw:
            result["raw_response"] = response
        return result

    workers = args.workers if client.supports_parallel else 1
    run_resumable(
        records,
        args.output,
        error_path,
        process,
        workers=workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
