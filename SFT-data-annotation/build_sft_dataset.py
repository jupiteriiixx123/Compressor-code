import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

"""
python build_sft_dataset.py \
  --normalized \
    processed_data/normalized/govreport_gao_train.jsonl \
    processed_data/normalized/govreport_crs_train.jsonl \
    processed_data/normalized/qasper_train.jsonl \
  --annotations annotation_outputs \
  --master-output training_data/master_annotations.jsonl \
  --sft-output training_data/sft_train.jsonl


master_annotations:保留evidence，用于人工分析审核
sft_train.jsonl:用于SFT，格式
{
  "id": "govreport_crs_RL33318",
  "dataset": "govreport",
  "subset": "crs",
  "messages": [
    {
      "role": "system",
      "content": "You are a long-context compression model. Compress the document into a coherent summary with expandable semantic information blocks. Output valid JSON with exactly two keys: \"summary\" and \"link\". Use ASCII citation anchors in the form [cite_n]. The summary length, citation count, block count, block length, and compression ratio must adapt freely to the document's information content."
    },
    {
      "role": "user",
      "content": "# Federal Cocaine Sentencing Disparity...\n\n[P0001] ..."
    },
    {
      "role": "assistant",
      "content": "{\"summary\":\"... [cite_1] ...\",\"link\":{\"cite_1\":\"...\"}}"
    }
  ]
}
"""


SYSTEM_PROMPT = (
    "You are a long-context compression model. "
    "Compress the document into a coherent summary with expandable "
    "semantic information blocks. "
    "Output valid JSON with exactly two keys: \"summary\" and \"link\". "
    "Use ASCII citation anchors in the form [cite_n]. "
    "The summary length, citation count, block count, block length, "
    "and compression ratio must adapt freely to the document's "
    "information content."
)

CITE_PATTERN = re.compile(r"\[cite_(\d+)\]")


# ============================================================
# Basic I/O
# ============================================================

def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {e}"
                )


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Load normalized mother dataset
# ============================================================

def load_normalized(
    normalized_files,
) -> Tuple[Dict[str, dict], list]:
    """
    Returns:
        records_by_id
        normalized_order

    normalized_order preserves the original JSONL ordering.
    """

    records = {}
    order = []

    for path in normalized_files:
        print(f"Loading normalized dataset: {path}")

        for record in read_jsonl(path):

            doc_id = record.get("id")

            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(
                    f"Normalized record has invalid id in {path}"
                )

            if doc_id in records:
                raise ValueError(
                    f"Duplicate normalized id: {doc_id}"
                )

            document = record.get("document")

            if not isinstance(document, str) or not document.strip():
                raise ValueError(
                    f"Normalized document is empty: {doc_id}"
                )

            records[doc_id] = record
            order.append(doc_id)

    return records, order


# ============================================================
# Annotation validation
# ============================================================

def validate_annotation(
    path: Path,
    annotation: dict,
    normalized: dict,
):
    if not isinstance(annotation, dict):
        raise ValueError(
            f"{path}: annotation is not a JSON object"
        )

    required = {
        "id",
        "summary",
        "link",
        "evidence",
    }

    missing = required - set(annotation.keys())

    if missing:
        raise ValueError(
            f"{path}: missing fields {sorted(missing)}"
        )

    doc_id = annotation["id"]

    # Filename must equal ID
    if path.stem != doc_id:
        raise ValueError(
            f"{path}: filename/id mismatch "
            f"({path.stem} != {doc_id})"
        )

    if doc_id not in normalized:
        raise ValueError(
            f"{path}: id not found in normalized data: {doc_id}"
        )

    summary = annotation["summary"]
    link = annotation["link"]
    evidence = annotation["evidence"]

    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(
            f"{path}: empty summary"
        )

    if not isinstance(link, dict):
        raise ValueError(
            f"{path}: link must be an object"
        )

    if not isinstance(evidence, dict):
        raise ValueError(
            f"{path}: evidence must be an object"
        )

    # --------------------------------------------------------
    # Citation validation
    # --------------------------------------------------------

    if "【cite_" in summary:
        raise ValueError(
            f"{path}: Chinese citation brackets detected"
        )

    citation_numbers = [
        int(x)
        for x in CITE_PATTERN.findall(summary)
    ]

    citation_keys = {
        f"cite_{n}"
        for n in citation_numbers
    }

    link_keys = set(link.keys())
    evidence_keys = set(evidence.keys())

    if citation_keys != link_keys:
        raise ValueError(
            f"{path}: summary citations != link keys"
        )

    if link_keys != evidence_keys:
        raise ValueError(
            f"{path}: link keys != evidence keys"
        )

    if citation_numbers:
        unique_numbers = sorted(set(citation_numbers))

        expected = list(
            range(1, max(unique_numbers) + 1)
        )

        if unique_numbers != expected:
            raise ValueError(
                f"{path}: non-consecutive citation numbering "
                f"{unique_numbers}"
            )

    # --------------------------------------------------------
    # Link validation
    # --------------------------------------------------------

    for cite, text in link.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{path}: empty link block: {cite}"
            )

    # --------------------------------------------------------
    # Evidence must reference real paragraph IDs
    # --------------------------------------------------------

    document = normalized[doc_id]["document"]

    valid_paragraph_ids = set(
        re.findall(r"\[(P\d+)\]", document)
    )

    for cite, paragraph_ids in evidence.items():

        if not isinstance(paragraph_ids, list):
            raise ValueError(
                f"{path}: evidence for {cite} is not a list"
            )

        for pid in paragraph_ids:

            if pid not in valid_paragraph_ids:
                raise ValueError(
                    f"{path}: nonexistent evidence paragraph "
                    f"{pid} in {cite}"
                )


# ============================================================
# Load annotation outputs
# ============================================================

def load_annotations(
    annotation_dir: Path,
    normalized: Dict[str, dict],
) -> Dict[str, dict]:

    annotations = {}

    annotation_files = sorted(
        annotation_dir.rglob("*.json")
    )

    print(
        f"Found {len(annotation_files)} annotation files."
    )

    for path in annotation_files:

        annotation = load_json(path)

        validate_annotation(
            path=path,
            annotation=annotation,
            normalized=normalized,
        )

        doc_id = annotation["id"]

        if doc_id in annotations:
            raise ValueError(
                f"Duplicate annotation id: {doc_id}"
            )

        annotations[doc_id] = annotation

    return annotations


# ============================================================
# Build master record
# ============================================================

def build_master_record(
    normalized: dict,
    annotation: dict,
) -> dict:

    return {
        "id": normalized["id"],
        "dataset": normalized["dataset"],
        "subset": normalized.get("subset"),
        "title": normalized.get("title"),
        "document": normalized["document"],

        "annotation": {
            "summary": annotation["summary"],
            "link": annotation["link"],
            "evidence": annotation["evidence"],
        },

        "metadata": normalized.get(
            "metadata",
            {},
        ),
    }


# ============================================================
# Build actual SFT record
# ============================================================

def build_sft_record(
    normalized: dict,
    annotation: dict,
) -> dict:

    # Student should NOT learn evidence.
    target = {
        "summary": annotation["summary"],
        "link": annotation["link"],
    }

    # Important:
    # assistant content is a JSON STRING because messages
    # content itself must be textual.
    assistant_content = json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return {
        "id": normalized["id"],
        "dataset": normalized["dataset"],
        "subset": normalized.get("subset"),

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": normalized["document"],
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Join normalized documents with completed Codex "
            "annotations and build master + SFT JSONL datasets."
        )
    )

    parser.add_argument(
        "--normalized",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more normalized JSONL files."
        ),
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help=(
            "Root directory containing annotation output JSON files."
        ),
    )

    parser.add_argument(
        "--master-output",
        type=Path,
        default=Path(
            "training_data/master_annotations.jsonl"
        ),
    )

    parser.add_argument(
        "--sft-output",
        type=Path,
        default=Path(
            "training_data/sft_train.jsonl"
        ),
    )

    parser.add_argument(
        "--require-all",
        action="store_true",
        help=(
            "Require every normalized document to have an "
            "annotation. Normally disabled because only a "
            "subset may be selected for SFT."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load normalized mother dataset
    # --------------------------------------------------------

    normalized, normalized_order = (
        load_normalized(args.normalized)
    )

    print()
    print(
        f"Normalized documents: {len(normalized)}"
    )

    # --------------------------------------------------------
    # Load + validate annotations
    # --------------------------------------------------------

    annotations = load_annotations(
        annotation_dir=args.annotations,
        normalized=normalized,
    )

    print(
        f"Valid annotations:    {len(annotations)}"
    )

    # --------------------------------------------------------
    # Optional strict mode
    # --------------------------------------------------------

    if args.require_all:

        missing = (
            set(normalized.keys())
            - set(annotations.keys())
        )

        if missing:
            raise RuntimeError(
                f"{len(missing)} normalized documents "
                f"have no annotation."
            )

    # --------------------------------------------------------
    # Preserve normalized ordering
    # --------------------------------------------------------

    selected_ids = [
        doc_id
        for doc_id in normalized_order
        if doc_id in annotations
    ]

    print(
        f"Training examples:    {len(selected_ids)}"
    )

    # --------------------------------------------------------
    # Master dataset
    # --------------------------------------------------------

    master_records = (
        build_master_record(
            normalized[doc_id],
            annotations[doc_id],
        )
        for doc_id in selected_ids
    )

    write_jsonl(
        args.master_output,
        master_records,
    )

    # --------------------------------------------------------
    # SFT dataset
    # --------------------------------------------------------

    sft_records = (
        build_sft_record(
            normalized[doc_id],
            annotations[doc_id],
        )
        for doc_id in selected_ids
    )

    write_jsonl(
        args.sft_output,
        sft_records,
    )

    print()
    print("=" * 64)
    print("Dataset build completed")
    print("=" * 64)
    print(
        f"Master: {args.master_output}"
    )
    print(
        f"SFT:    {args.sft_output}"
    )
    print(
        f"Rows:   {len(selected_ids)}"
    )


if __name__ == "__main__":
    main()