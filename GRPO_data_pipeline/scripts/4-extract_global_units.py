#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_data.io_utils import extract_first_json_object, normalize_ws, read_jsonl
from stage2_data.llm import add_llm_args, build_client
from stage2_data.prompts import GLOBAL_SYSTEM, global_user_prompt
from stage2_data.runner import run_resumable
from stage2_data.schema import parse_units_payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--articles", required=True)
    p.add_argument("--chunks", required=True)
    p.add_argument("--local-units", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--errors", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    add_llm_args(p)
    args = p.parse_args()

    client = build_client(args)

    local_by_context = collections.defaultdict(list)
    for row in read_jsonl(args.local_units):
        local_by_context[row["context_id"]].extend(row.get("units", []))

    chunks_by_context = collections.defaultdict(list)
    for row in read_jsonl(args.chunks):
        chunks_by_context[row["context_id"]].append(row)

    articles = list(read_jsonl(args.articles))
    if args.limit is not None:
        articles = articles[:args.limit]

    error_path = args.errors or str(Path(args.output).with_suffix(".errors.jsonl"))

    def process(article):
        response = client.generate(
            GLOBAL_SYSTEM,
            global_user_prompt(article["context"], local_by_context.get(article["id"], [])),
        )
        payload = extract_first_json_object(response)
        units = parse_units_payload(payload)

        source_chunks = chunks_by_context.get(article["id"], [])
        for unit in units:
            locations = []
            for ev in unit["evidence"]:
                ev_norm = normalize_ws(ev)
                locations.append([
                    c["chunk_id"]
                    for c in source_chunks
                    if ev_norm and ev_norm in normalize_ws(c["text"])
                ])
            unit["evidence_chunk_matches"] = locations

        result = {"id": article["id"], "context_id": article["id"], "units": units}
        if args.keep_raw:
            result["raw_response"] = response
        return result

    workers = args.workers if client.supports_parallel else 1
    run_resumable(
        articles, args.output, error_path, process,
        workers=workers, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
