import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


"""
启动5次codex，每次6条数据
python run_annotation_batches.py \
  --batch-size 6 \
  --max-batches 60

一口气跑完
python run_annotation_batches.py \
  --batch-size 6 \
  --max-batches 0
"""


HTTP_PROXY = "http://127.0.0.1:6789"
ALL_PROXY = "socks5://127.0.0.1:6789"

REQUIRED_OUTPUT_KEYS = {
    "id",
    "summary",
    "link",
    "evidence",
}

CITE_PATTERN = re.compile(r"\[cite_(\d+)\]")


def check_proxy(host="127.0.0.1", port=6789, timeout=2):
    """
    Check whether the local proxy port is reachable.
    """

    try:
        with socket.create_connection(
            (host, port),
            timeout=timeout,
        ):
            return True
    except OSError:
        return False

# ============================================================
# Basic JSON utilities
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_expected_output(
    input_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """
    Preserve relative path.

    annotation_inputs/qasper/qasper_123.json
        ->
    annotation_outputs/qasper/qasper_123.json
    """
    relative = input_path.relative_to(input_root)
    return output_root / relative


# ============================================================
# Input validation
# ============================================================

def validate_input_file(path: Path):
    """
    Validate that input filename is strictly <id>.json.
    """

    try:
        data = load_json(path)
    except Exception as e:
        return False, f"cannot parse input JSON: {e}"

    doc_id = data.get("id")

    if not isinstance(doc_id, str) or not doc_id:
        return False, "missing or invalid id"

    if path.stem != doc_id:
        return False, (
            f"filename/id mismatch: "
            f"filename={path.stem}, id={doc_id}"
        )

    if not isinstance(data.get("document"), str):
        return False, "missing document"

    return True, None


# ============================================================
# Output/checkpoint validation
# ============================================================

def validate_completed_output(
    input_path: Path,
    output_path: Path,
):
    """
    A file counts as a completed checkpoint ONLY if it passes
    structural validation.

    This does NOT judge annotation quality or impose any
    compression length / citation count.
    """

    if not output_path.exists():
        return False, "output missing"

    try:
        inp = load_json(input_path)
        out = load_json(output_path)
    except Exception as e:
        return False, f"JSON parse error: {e}"

    # --------------------------------------------------------
    # Basic schema
    # --------------------------------------------------------

    if not isinstance(out, dict):
        return False, "output is not a JSON object"

    missing = REQUIRED_OUTPUT_KEYS - set(out.keys())

    if missing:
        return False, f"missing keys: {sorted(missing)}"

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if out["id"] != inp["id"]:
        return False, (
            f"id mismatch: expected {inp['id']}, "
            f"got {out['id']}"
        )

    if output_path.stem != inp["id"]:
        return False, "output filename does not match id"

    # --------------------------------------------------------
    # Field types
    # --------------------------------------------------------

    summary = out["summary"]
    link = out["link"]
    evidence = out["evidence"]

    if not isinstance(summary, str) or not summary.strip():
        return False, "summary is empty"

    if not isinstance(link, dict):
        return False, "link is not an object"

    if not isinstance(evidence, dict):
        return False, "evidence is not an object"

    # --------------------------------------------------------
    # Citation format
    # --------------------------------------------------------

    # Explicitly reject Chinese-style citation brackets
    if "【cite_" in summary:
        return False, "contains Chinese citation brackets"

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

    # Summary/link/evidence must correspond exactly
    if citation_keys != link_keys:
        return False, (
            "summary citations and link keys do not match"
        )

    if link_keys != evidence_keys:
        return False, (
            "link keys and evidence keys do not match"
        )

    # --------------------------------------------------------
    # Consecutive numbering
    # --------------------------------------------------------

    if citation_numbers:
        unique_numbers = sorted(set(citation_numbers))

        expected = list(
            range(1, max(unique_numbers) + 1)
        )

        if unique_numbers != expected:
            return False, (
                f"citation numbering is not consecutive: "
                f"{unique_numbers}"
            )

    # --------------------------------------------------------
    # Link contents
    # --------------------------------------------------------

    for key, value in link.items():
        if not isinstance(value, str) or not value.strip():
            return False, f"{key} has empty link content"

    # --------------------------------------------------------
    # Evidence paragraph IDs must exist in source
    # --------------------------------------------------------

    document = inp["document"]

    valid_pids = set(
        re.findall(r"\[(P\d+)\]", document)
    )

    for cite_key, pids in evidence.items():

        if not isinstance(pids, list):
            return False, (
                f"{cite_key} evidence is not a list"
            )

        for pid in pids:

            if not isinstance(pid, str):
                return False, (
                    f"{cite_key} contains non-string evidence"
                )

            if pid not in valid_pids:
                return False, (
                    f"{cite_key} references nonexistent "
                    f"paragraph {pid}"
                )

    return True, None


# ============================================================
# Discover pending inputs
# ============================================================

def discover_tasks(
    input_root: Path,
    output_root: Path,
):
    """
    Returns:
        completed
        pending
        invalid_inputs
    """

    input_files = sorted(
        input_root.rglob("*.json")
    )

    completed = []
    pending = []
    invalid_inputs = []

    for input_path in input_files:

        input_valid, reason = validate_input_file(
            input_path
        )

        if not input_valid:
            invalid_inputs.append(
                (input_path, reason)
            )
            continue

        output_path = get_expected_output(
            input_path,
            input_root,
            output_root,
        )

        complete, reason = validate_completed_output(
            input_path,
            output_path,
        )

        if complete:
            completed.append(
                (input_path, output_path)
            )
        else:
            pending.append(
                (input_path, output_path)
            )

    return completed, pending, invalid_inputs


# ============================================================
# Build a very explicit Codex prompt for ONE batch
# ============================================================

def build_codex_prompt(
    project_root: Path,
    tasks,
):
    lines = []

    lines.append(
        "Read and follow AGENTS.md and "
        "annotation_guideline.md strictly."
    )

    lines.append("")

    lines.append(
        f"Process exactly the following {len(tasks)} "
        "annotation tasks and no other documents."
    )

    lines.append("")

    lines.append(
        "Every annotation must be produced directly by "
        "your own language-model reasoning after reading "
        "the full input document."
    )

    lines.append("")

    lines.append(
        "Do not search for or use local language models, "
        "external models, summarization libraries, heuristic "
        "compressors, extractive templates, or scripts that "
        "generate semantic annotation content."
    )

    lines.append("")

    lines.append(
        "Do not inspect completed annotation outputs for "
        "style, citation count, summary length, or structure."
    )

    lines.append("")

    lines.append(
        "Treat every document independently. There is no "
        "preferred summary length, citation count, block "
        "count, block length, or compression ratio."
    )

    lines.append("")

    lines.append(
        "Citation placement must follow the semantic-anchor "
        "rules in annotation_guideline.md. Do not force a "
        "one-sentence = one-citation pattern."
    )

    lines.append("")

    lines.append(
        "Process the documents ONE AT A TIME."
    )

    lines.append(
        "After finishing each document, immediately write "
        "its JSON output to the specified output path before "
        "starting the next document."
    )

    lines.append("")

    lines.append("Tasks:")

    for i, (input_path, output_path) in enumerate(
        tasks,
        start=1,
    ):
        input_rel = input_path.relative_to(project_root)
        output_rel = output_path.relative_to(project_root)

        lines.append("")
        lines.append(f"{i}.")
        lines.append(
            f"   INPUT:  {input_rel}"
        )
        lines.append(
            f"   OUTPUT: {output_rel}"
        )

    lines.append("")

    lines.append(
        "Create parent output directories if needed."
    )

    lines.append(
        "Do not modify any input document."
    )

    lines.append(
        "Do not process additional files after these tasks "
        "are complete."
    )

    lines.append(
        "Do not merely describe what you would do. "
        "Actually read, annotate, validate, and save each "
        "listed document."
    )

    return "\n".join(lines)


# ============================================================
# Run one fresh Codex CLI process
# ============================================================

def run_codex_batch(
    project_root: Path,
    tasks,
):
    prompt = build_codex_prompt(
        project_root,
        tasks,
    )

    print()
    print("=" * 72)
    print(
        f"Starting fresh Codex session "
        f"for {len(tasks)} documents"
    )
    print("=" * 72)

    for input_path, _ in tasks:
        print(
            "  -",
            input_path.relative_to(project_root)
        )

    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Check local proxy before launching Codex
    # --------------------------------------------------------

    if not check_proxy():
        print(
            "[STOP] Proxy is not reachable at "
            "127.0.0.1:6789"
        )
        print(
            "Please start your proxy service before "
            "continuing."
        )
        return 100

    # --------------------------------------------------------
    # Prepare environment for the Codex child process
    # --------------------------------------------------------

    env = os.environ.copy()

    # Lowercase forms
    env["https_proxy"] = HTTP_PROXY
    env["http_proxy"] = HTTP_PROXY
    env["all_proxy"] = ALL_PROXY

    # Uppercase forms as well, for compatibility
    env["HTTPS_PROXY"] = HTTP_PROXY
    env["HTTP_PROXY"] = HTTP_PROXY
    env["ALL_PROXY"] = ALL_PROXY

    print("Proxy configuration:")
    print(f"  http_proxy  = {HTTP_PROXY}")
    print(f"  https_proxy = {HTTP_PROXY}")
    print(f"  all_proxy   = {ALL_PROXY}")
    print()

    # --------------------------------------------------------
    # Start a completely fresh Codex CLI process
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "codex",
            "exec",
            "--skip-git-repo-check",
            prompt,
        ],
        cwd=project_root,
        env=env,
    )

    return result.returncode


# ============================================================
# Main scheduler
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run long-document annotation through Codex CLI "
            "in small fresh-context batches."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing AGENTS.md.",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("annotation_inputs"),
        help="Input directory relative to project root.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotation_outputs"),
        help="Output directory relative to project root.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Documents per fresh Codex session.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help=(
            "Maximum batches in this run. "
            "Use 0 to continue until no pending files remain."
        ),
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Seconds to pause between Codex sessions.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.max_batches < 0:
        raise ValueError(
            "--max-batches must be >= 0"
        )

    project_root = args.project_root.resolve()

    input_root = (
        project_root / args.input_dir
    ).resolve()

    output_root = (
        project_root / args.output_dir
    ).resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_root}"
        )

    if not (project_root / "AGENTS.md").exists():
        print(
            "[WARNING] AGENTS.md not found in project root."
        )

    if not (
        project_root / "annotation_guideline.md"
    ).exists():
        print(
            "[WARNING] annotation_guideline.md "
            "not found in project root."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    batches_run = 0

    try:

        while True:

            completed, pending, invalid_inputs = (
                discover_tasks(
                    input_root,
                    output_root,
                )
            )

            print()
            print("=" * 72)
            print("Annotation status")
            print("=" * 72)

            print(
                f"Completed:      {len(completed)}"
            )
            print(
                f"Pending:        {len(pending)}"
            )
            print(
                f"Invalid inputs: {len(invalid_inputs)}"
            )

            if invalid_inputs:
                print()
                print("Invalid input files:")

                for path, reason in invalid_inputs:
                    print(
                        f"  {path}: {reason}"
                    )

                print()
                print(
                    "Fix invalid inputs before continuing."
                )

                sys.exit(1)

            if not pending:
                print()
                print("All annotations completed.")
                break

            # Stop if requested number of batches has run.
            if (
                args.max_batches != 0
                and batches_run >= args.max_batches
            ):
                print()
                print(
                    "Reached --max-batches limit."
                )
                break

            batch = pending[:args.batch_size]

            returncode = run_codex_batch(
                project_root,
                batch,
            )

            batches_run += 1

            if returncode != 0:
                print()
                print(
                    f"[STOP] Codex exited with "
                    f"code {returncode}."
                )
                print(
                    "Completed outputs remain intact. "
                    "Run the scheduler again to resume."
                )
                sys.exit(returncode)

            # ------------------------------------------------
            # Immediately validate this batch after Codex exits
            # ------------------------------------------------

            print()
            print("Validating batch outputs...")

            failures = []

            for input_path, output_path in batch:

                valid, reason = validate_completed_output(
                    input_path,
                    output_path,
                )

                if valid:
                    print(
                        f"[PASS] {output_path.name}"
                    )
                else:
                    print(
                        f"[FAIL] {output_path.name}: "
                        f"{reason}"
                    )

                    failures.append(
                        (input_path, reason)
                    )

            # Important:
            # stop instead of endlessly retrying malformed output.
            if failures:
                print()
                print(
                    "[STOP] One or more annotations failed "
                    "validation."
                )
                print(
                    "Valid files were kept. Invalid/missing "
                    "files remain pending."
                )
                print(
                    "Inspect the failed case before resuming."
                )
                sys.exit(2)

            print()
            print(
                f"Batch {batches_run} completed successfully."
            )

            if args.pause > 0:
                time.sleep(args.pause)

    except KeyboardInterrupt:
        print()
        print()
        print("[STOP] Interrupted by user.")
        print(
            "Any valid outputs already written are preserved."
        )
        print(
            "Run the same command again to resume."
        )
        sys.exit(130)


if __name__ == "__main__":
    main()