from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_ids(path: str | Path, id_key: str = "id") -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        value = row.get(id_key)
        if value is not None:
            done.add(str(value))
    return done


_ws_re = re.compile(r"\s+")


def normalize_ws(text: Any) -> str:
    return _ws_re.sub(" ", str(text)).strip()


def normalize_claim(text: Any) -> str:
    text = normalize_ws(text).casefold()
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.rstrip(" .")


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text, count=1)
    return text.strip()


def extract_json_value(text: str) -> Any:
    """Parse the first JSON value from model output.

    Supports both top-level arrays and objects. Direct parsing is attempted first;
    then a tolerant scan is used for outputs that accidentally contain a prefix.
    """
    text = _strip_markdown_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not locate a valid JSON value in model output.")


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """Backward-compatible helper for global extraction / dedup scripts."""
    value = extract_json_value(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a top-level JSON object in model output.")
    return value


def answer_letter(index_1based: int | str | None) -> str | None:
    if index_1based is None:
        return None
    try:
        idx = int(index_1based)
    except (TypeError, ValueError):
        return None
    if 1 <= idx <= 26:
        return chr(ord("A") + idx - 1)
    return None


class ModelOutputError(ValueError):
    """Parsing/validation error that retains the raw model output for debugging."""

    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response
