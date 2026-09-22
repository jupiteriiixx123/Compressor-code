from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List

from tqdm import tqdm

from .io_utils import append_jsonl, load_completed_ids


def _error_row(record: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": record.get("id"),
        "error": repr(exc),
    }
    raw_response = getattr(exc, "raw_response", None)
    if raw_response is not None:
        row["raw_response"] = raw_response
    return row


def run_resumable(
    records: List[Dict[str, Any]],
    output_path: str,
    error_path: str,
    process_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    workers: int = 1,
    overwrite: bool = False,
) -> None:
    out = Path(output_path)
    err = Path(error_path)

    if overwrite:
        if out.exists():
            out.unlink()
        if err.exists():
            err.unlink()

    done = load_completed_ids(out)
    pending = [r for r in records if str(r["id"]) not in done]

    if workers <= 1:
        for record in tqdm(pending, desc="processing"):
            try:
                append_jsonl(out, process_fn(record))
            except Exception as e:
                append_jsonl(err, _error_row(record, e))
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_fn, r): r for r in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="processing"):
            record = futures[fut]
            try:
                append_jsonl(out, fut.result())
            except Exception as e:
                append_jsonl(err, _error_row(record, e))
