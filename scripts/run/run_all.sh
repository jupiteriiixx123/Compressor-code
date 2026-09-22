#!/usr/bin/env bash
set -euo pipefail

# Run all benchmark inference except LongMemEval.
# Group 1 (LoCoMo): Stage-0/1/2 compression + full/stage0/1/2 reader.
# Group 2 (GovReport, MultiNews): Stage-0/1/2 compression only.
# Existing JSONL outputs are resumed by the Python runners unless OVERWRITE=1.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/hh_hh/LoRA/condense}"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMPRESSOR="$PROJECT_ROOT/experience/scripts/run/run_compressor.py"
READER="$PROJECT_ROOT/experience/scripts/run/run_reader.py"

STAGE0_MODEL="${STAGE0_MODEL:-/public/home/hh_hh/LoRA/models/Qwen3-4B}"
STAGE1_MODEL="${STAGE1_MODEL:-$PROJECT_ROOT/outputs/sft_full_600_v2_7ep}"
STAGE2_ADAPTER="${STAGE2_ADAPTER:-$PROJECT_ROOT/outputs/stage2_gdpo_cite}"
READER_MODEL="${READER_MODEL:-/public/home/hh_hh/LoRA/models/Qwen3-8B}"

SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
STAGE0_DEVICE="${STAGE0_DEVICE:-cuda:2}"
STAGE1_DEVICE="${STAGE1_DEVICE:-cuda:0}"
STAGE2_DEVICE="${STAGE2_DEVICE:-cuda:2}"
READER_DEVICE="${READER_DEVICE:-cuda:0}"
OVERWRITE="${OVERWRITE:-0}"
MAX_COMPRESS_SAMPLES="${MAX_COMPRESS_SAMPLES:-}"
MAX_READER_SOURCES="${MAX_READER_SOURCES:-}"

LOCOMO_DIR="$PROJECT_ROOT/experience/processed/group1/locomo"
GOV_DIR="$PROJECT_ROOT/experience/processed/group2/gov_report"
MULTI_DIR="$PROJECT_ROOT/experience/processed/group2/multi_news"
COMP_ROOT="$PROJECT_ROOT/experience/compressions"
PRED_ROOT="$PROJECT_ROOT/experience/predictions"

mkdir -p \
  "$COMP_ROOT/stage0/group1" "$COMP_ROOT/stage1/group1" "$COMP_ROOT/stage2/group1" \
  "$COMP_ROOT/stage0/group2" "$COMP_ROOT/stage1/group2" "$COMP_ROOT/stage2/group2" \
  "$PRED_ROOT/full_context/group1" "$PRED_ROOT/stage0/group1" \
  "$PRED_ROOT/stage1/group1" "$PRED_ROOT/stage2/group1"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: required file not found: $1" >&2; exit 1; }
}

require_file "$COMPRESSOR"
require_file "$READER"
require_file "$LOCOMO_DIR/compressor_input.jsonl"
require_file "$LOCOMO_DIR/eval_meta.jsonl"
require_file "$GOV_DIR/compressor_input.jsonl"
require_file "$MULTI_DIR/compressor_input.jsonl"

COMMON_COMP_ARGS=(--dtype "$DTYPE" --seed "$SEED" --no-thinking)
COMMON_READER_ARGS=(--dtype "$DTYPE" --seed "$SEED" --no-thinking)

if [[ "$OVERWRITE" == "1" ]]; then
  COMMON_COMP_ARGS+=(--overwrite)
  COMMON_READER_ARGS+=(--overwrite)
fi
if [[ -n "$MAX_COMPRESS_SAMPLES" ]]; then
  COMMON_COMP_ARGS+=(--max-samples "$MAX_COMPRESS_SAMPLES")
fi
if [[ -n "$MAX_READER_SOURCES" ]]; then
  COMMON_READER_ARGS+=(--max-sources "$MAX_READER_SOURCES")
fi

run_cmd() {
  local title="$1"; shift
  echo
  echo "======================================================================"
  echo "$title"
  echo "======================================================================"
  "$@"
}

run_compressor() {
  local title="$1" input="$2" output="$3" base_model="$4" device="$5" adapter="${6:-}"
  local cmd=(
    "$PYTHON_BIN" "$COMPRESSOR"
    --input "$input"
    --output "$output"
    --base-model "$base_model"
    --device "$device"
    "${COMMON_COMP_ARGS[@]}"
  )
  [[ -n "$adapter" ]] && cmd+=(--adapter "$adapter")
  run_cmd "$title" "${cmd[@]}"
}

run_reader() {
  local title="$1" memory_input="$2" eval_meta="$3" output="$4" memory_mode="$5"
  local cmd=(
    "$PYTHON_BIN" "$READER"
    --memory-input "$memory_input"
    --eval-meta "$eval_meta"
    --output "$output"
    --memory-mode "$memory_mode"
    --reader-model "$READER_MODEL"
    --device "$READER_DEVICE"
    "${COMMON_READER_ARGS[@]}"
  )
  run_cmd "$title" "${cmd[@]}"
}

echo "======================================================================"
echo "BENCHMARK INFERENCE: ALL EXCEPT LONGMEMEVAL"
echo "======================================================================"
echo "Project root : $PROJECT_ROOT"
echo "Seed         : $SEED"
echo "Stage-0 GPU  : $STAGE0_DEVICE"
echo "Stage-1 GPU  : $STAGE1_DEVICE"
echo "Stage-2 GPU  : $STAGE2_DEVICE"
echo "Reader GPU   : $READER_DEVICE"
echo "Overwrite    : $OVERWRITE"
echo "======================================================================"

# 1-3. LoCoMo compression
run_compressor "[1/13] LoCoMo | Stage-0 compression" \
  "$LOCOMO_DIR/compressor_input.jsonl" "$COMP_ROOT/stage0/group1/locomo.jsonl" \
  "$STAGE0_MODEL" "$STAGE0_DEVICE"
run_compressor "[2/13] LoCoMo | Stage-1 compression" \
  "$LOCOMO_DIR/compressor_input.jsonl" "$COMP_ROOT/stage1/group1/locomo.jsonl" \
  "$STAGE1_MODEL" "$STAGE1_DEVICE"
run_compressor "[3/13] LoCoMo | Stage-2 compression" \
  "$LOCOMO_DIR/compressor_input.jsonl" "$COMP_ROOT/stage2/group1/locomo.jsonl" \
  "$STAGE1_MODEL" "$STAGE2_DEVICE" "$STAGE2_ADAPTER"

# 4-6. GovReport compression
run_compressor "[4/13] GovReport | Stage-0 compression" \
  "$GOV_DIR/compressor_input.jsonl" "$COMP_ROOT/stage0/group2/gov_report.jsonl" \
  "$STAGE0_MODEL" "$STAGE0_DEVICE"
run_compressor "[5/13] GovReport | Stage-1 compression" \
  "$GOV_DIR/compressor_input.jsonl" "$COMP_ROOT/stage1/group2/gov_report.jsonl" \
  "$STAGE1_MODEL" "$STAGE1_DEVICE"
run_compressor "[6/13] GovReport | Stage-2 compression" \
  "$GOV_DIR/compressor_input.jsonl" "$COMP_ROOT/stage2/group2/gov_report.jsonl" \
  "$STAGE1_MODEL" "$STAGE2_DEVICE" "$STAGE2_ADAPTER"

# 7-9. MultiNews compression
run_compressor "[7/13] MultiNews | Stage-0 compression" \
  "$MULTI_DIR/compressor_input.jsonl" "$COMP_ROOT/stage0/group2/multi_news.jsonl" \
  "$STAGE0_MODEL" "$STAGE0_DEVICE"
run_compressor "[8/13] MultiNews | Stage-1 compression" \
  "$MULTI_DIR/compressor_input.jsonl" "$COMP_ROOT/stage1/group2/multi_news.jsonl" \
  "$STAGE1_MODEL" "$STAGE1_DEVICE"
run_compressor "[9/13] MultiNews | Stage-2 compression" \
  "$MULTI_DIR/compressor_input.jsonl" "$COMP_ROOT/stage2/group2/multi_news.jsonl" \
  "$STAGE1_MODEL" "$STAGE2_DEVICE" "$STAGE2_ADAPTER"

# 10-13. LoCoMo reader
run_reader "[10/13] LoCoMo | Full-context reader" \
  "$LOCOMO_DIR/compressor_input.jsonl" "$LOCOMO_DIR/eval_meta.jsonl" \
  "$PRED_ROOT/full_context/group1/locomo.jsonl" "full_context"
run_reader "[11/13] LoCoMo | Stage-0 reader" \
  "$COMP_ROOT/stage0/group1/locomo.jsonl" "$LOCOMO_DIR/eval_meta.jsonl" \
  "$PRED_ROOT/stage0/group1/locomo.jsonl" "compressed"
run_reader "[12/13] LoCoMo | Stage-1 reader" \
  "$COMP_ROOT/stage1/group1/locomo.jsonl" "$LOCOMO_DIR/eval_meta.jsonl" \
  "$PRED_ROOT/stage1/group1/locomo.jsonl" "compressed"
run_reader "[13/13] LoCoMo | Stage-2 reader" \
  "$COMP_ROOT/stage2/group1/locomo.jsonl" "$LOCOMO_DIR/eval_meta.jsonl" \
  "$PRED_ROOT/stage2/group1/locomo.jsonl" "compressed"

echo
echo "======================================================================"
echo "ALL NON-LONGMEMEVAL INFERENCE FINISHED"
echo "======================================================================"
echo "Compression outputs:"
echo "  $COMP_ROOT/stage{0,1,2}/group1/locomo.jsonl"
echo "  $COMP_ROOT/stage{0,1,2}/group2/gov_report.jsonl"
echo "  $COMP_ROOT/stage{0,1,2}/group2/multi_news.jsonl"
echo "Reader outputs:"
echo "  $PRED_ROOT/full_context/group1/locomo.jsonl"
echo "  $PRED_ROOT/stage{0,1,2}/group1/locomo.jsonl"
echo "Next step: evaluation."