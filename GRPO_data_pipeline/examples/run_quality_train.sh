#!/usr/bin/env bash
set -euo pipefail

RAW="${RAW:-/public/home/hh_hh/LoRA/condense/benchmarks/quality/QuALITY.v1.0.1.htmlstripped.train}"
WORK="${WORK:-/public/home/hh_hh/LoRA/condense/training_data/stage2/quality_work/train}"
FINAL="${FINAL:-/public/home/hh_hh/LoRA/condense/training_data/stage2/quality}"
POLICY_TOKENIZER="${POLICY_TOKENIZER:-/public/home/hh_hh/LoRA/models/Qwen3-4B}"

EXTRACTOR_TOKENIZER="${EXTRACTOR_TOKENIZER:-/path/to/Qwen-32B-Instruct}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-/path/to/Qwen-32B-Instruct}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
WORKERS="${WORKERS:-4}"

mkdir -p "$WORK" "$FINAL"

python scripts/01_prepare_quality.py \
  --input "$RAW" \
  --split train \
  --output "$WORK/articles.jsonl" \
  --tokenizer "$POLICY_TOKENIZER"

python scripts/02_chunk_contexts.py \
  --input "$WORK/articles.jsonl" \
  --output "$WORK/chunks.jsonl" \
  --tokenizer "$EXTRACTOR_TOKENIZER" \
  --chunk-tokens 1600 \
  --overlap-tokens 150

python scripts/03_extract_local_units.py \
  --input "$WORK/chunks.jsonl" \
  --output "$WORK/local_units.jsonl" \
  --backend openai_compatible \
  --base-url "$BASE_URL" \
  --model "$EXTRACTOR_MODEL" \
  --workers "$WORKERS" \
  --max-new-tokens 4096

python scripts/04_extract_global_units.py \
  --articles "$WORK/articles.jsonl" \
  --chunks "$WORK/chunks.jsonl" \
  --local-units "$WORK/local_units.jsonl" \
  --output "$WORK/global_units.jsonl" \
  --backend openai_compatible \
  --base-url "$BASE_URL" \
  --model "$EXTRACTOR_MODEL" \
  --workers "$WORKERS" \
  --max-new-tokens 4096

python scripts/05_merge_units.py \
  --articles "$WORK/articles.jsonl" \
  --local-units "$WORK/local_units.jsonl" \
  --global-units "$WORK/global_units.jsonl" \
  --output "$WORK/merged_units.jsonl"

python scripts/06_semantic_dedup.py \
  --input "$WORK/merged_units.jsonl" \
  --output "$WORK/final_units.jsonl" \
  --backend openai_compatible \
  --base-url "$BASE_URL" \
  --model "$EXTRACTOR_MODEL" \
  --workers "$WORKERS" \
  --max-new-tokens 4096

python scripts/07_build_stage2_dataset.py \
  --articles "$WORK/articles.jsonl" \
  --final-units "$WORK/final_units.jsonl" \
  --split train \
  --output-dir "$FINAL"

python scripts/08_validate_stage2_dataset.py \
  --input "$FINAL/stage2_train.jsonl" \
  --report "$FINAL/train_report.json"
