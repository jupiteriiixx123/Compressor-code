#!/usr/bin/env bash
set -euo pipefail

# Run LongMemEval benchmark inference.
# Compressor: Stage-0/1/2 all use --long-context.
# Reader: only Full-context uses --long-context; Stage-0/1/2 readers do NOT.

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/hh_hh/LoRA/condense}"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMPRESSOR="$PROJECT_ROOT/experience/scripts/run/run_compressor.py"
READER="$PROJECT_ROOT/experience/scripts/run/run_reader.py"

STAGE0_MODEL="${STAGE0_MODEL:-/public/home/hh_hh/LoRA/models/Qwen3-4B}"
STAGE1_MODEL="${STAGE1_MODEL:-/public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep}"
STAGE2_ADAPTER="${STAGE2_ADAPTER:-/public/home/hh_hh/LoRA/condense/outputs/stage2_gdpo_cite}"
READER_MODEL="${READER_MODEL:-/public/home/hh_hh/LoRA/models/Qwen3-8B}"

SEED="${SEED:-42}"
STAGE0_DEVICE="${STAGE0_DEVICE:-cuda:2}"
STAGE1_DEVICE="${STAGE1_DEVICE:-cuda:2}"
STAGE2_DEVICE="${STAGE2_DEVICE:-cuda:2}"
READER_DEVICE="${READER_DEVICE:-cuda:2}"

LONGMEM_DIR="$PROJECT_ROOT/experience/processed/group1/longmemeval"
COMP_ROOT="$PROJECT_ROOT/experience/compressions"
PRED_ROOT="$PROJECT_ROOT/experience/predictions"

mkdir -p \
  "$COMP_ROOT/stage0/group1" "$COMP_ROOT/stage1/group1" "$COMP_ROOT/stage2/group1" \
  "$PRED_ROOT/full_context/group1" "$PRED_ROOT/stage0/group1" \
  "$PRED_ROOT/stage1/group1" "$PRED_ROOT/stage2/group1"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: required file not found: $1" >&2; exit 1; }
}

require_file "$COMPRESSOR"
require_file "$READER"
require_file "$LONGMEM_DIR/compressor_input.jsonl"
require_file "$LONGMEM_DIR/eval_meta.jsonl"

run_cmd() {
  local title="$1"; shift
  echo
  echo "======================================================================"
  echo "$title"
  echo "======================================================================"
  "$@"
}

# --- Compressor: all three stages use --long-context ---
run_compressor() {
  local title="$1" input="$2" output="$3" base_model="$4" device="$5" adapter="${6:-}"
  local cmd=(
    "$PYTHON_BIN" "$COMPRESSOR"
    --input "$input"
    --output "$output"
    --base-model "$base_model"
    --device "$device"
    --seed "$SEED"
    --long-context
  )
  [[ -n "$adapter" ]] && cmd+=(--adapter "$adapter")
  run_cmd "$title" "${cmd[@]}"
}

# --- Reader: --long-context only for full_context ---
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
    --seed "$SEED"
  )
  [[ "$memory_mode" == "full_context" ]] && cmd+=(--long-context)
  run_cmd "$title" "${cmd[@]}"
}

echo "======================================================================"
echo "BENCHMARK INFERENCE: LONGMEMEVAL"
echo "======================================================================"
echo "Project root : $PROJECT_ROOT"
echo "Seed         : $SEED"
echo "Stage-0 GPU  : $STAGE0_DEVICE"
echo "Stage-1 GPU  : $STAGE1_DEVICE"
echo "Stage-2 GPU  : $STAGE2_DEVICE"
echo "Reader GPU   : $READER_DEVICE"
echo "======================================================================"

# 1-3. LongMemEval compression
run_compressor "[1/7] LongMemEval | Stage-0 compression" \
  "$LONGMEM_DIR/compressor_input.jsonl" "$COMP_ROOT/stage0/group1/longmemeval.jsonl" \
  "$STAGE0_MODEL" "$STAGE0_DEVICE"
run_compressor "[2/7] LongMemEval | Stage-1 compression" \
  "$LONGMEM_DIR/compressor_input.jsonl" "$COMP_ROOT/stage1/group1/longmemeval.jsonl" \
  "$STAGE1_MODEL" "$STAGE1_DEVICE"
run_compressor "[3/7] LongMemEval | Stage-2 compression" \
  "$LONGMEM_DIR/compressor_input.jsonl" "$COMP_ROOT/stage2/group1/longmemeval.jsonl" \
  "$STAGE1_MODEL" "$STAGE2_DEVICE" "$STAGE2_ADAPTER"

# 4-7. LongMemEval reader
run_reader "[4/7] LongMemEval | Full-context reader" \
  "$LONGMEM_DIR/compressor_input.jsonl" "$LONGMEM_DIR/eval_meta.jsonl" \
  "$PRED_ROOT/full_context/group1/longmemeval.jsonl" "full_context"
run_reader "[5/7] LongMemEval | Stage-0 reader" \
  "$COMP_ROOT/stage0/group1/longmemeval.jsonl" "$LONGMEM_DIR/eval_meta.jsonl" \
  "$PRED_ROOT/stage0/group1/longmemeval.jsonl" "compressed"
run_reader "[6/7] LongMemEval | Stage-1 reader" \
  "$COMP_ROOT/stage1/group1/longmemeval.jsonl" "$LONGMEM_DIR/eval_meta.jsonl" \
  "$PRED_ROOT/stage1/group1/longmemeval.jsonl" "compressed"
run_reader "[7/7] LongMemEval | Stage-2 reader" \
  "$COMP_ROOT/stage2/group1/longmemeval.jsonl" "$LONGMEM_DIR/eval_meta.jsonl" \
  "$PRED_ROOT/stage2/group1/longmemeval.jsonl" "compressed"

echo
echo "======================================================================"
echo "LONGMEMEVAL INFERENCE FINISHED"
echo "======================================================================"
echo "Compression outputs:"
echo "  $COMP_ROOT/stage{0,1,2}/group1/longmemeval.jsonl"
echo "Reader outputs:"
echo "  $PRED_ROOT/full_context/group1/longmemeval.jsonl"
echo "  $PRED_ROOT/stage{0,1,2}/group1/longmemeval.jsonl"
echo "Next step: evaluation."