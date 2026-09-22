# LoCoMo compressor
## Stage-0
```bash
python experience/scripts/run/run_compressor.py \
--input experience/processed/group1/locomo/compressor_input.jsonl \
--output experience/compressions/stage0/group1/locomo.jsonl \
--base-model /public/home/hh_hh/LoRA/models/Qwen3-4B \
--device cuda:2 \
--seed 42
```

## Stage-1
```bash
python experience/scripts/run/run_compressor.py \
  --input experience/processed/group1/locomo/compressor_input.jsonl \
  --output experience/compressions/stage1/group1/locomo.jsonl \
  --base-model /public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep \
  --device cuda:0 \
  --seed 42
```
## Stage-2
```bash
python experience/scripts/run/run_compressor.py \
  --input experience/processed/group1/locomo/compressor_input.jsonl \
  --output experience/compressions/stage2/group1/locomo.jsonl \
  --base-model /public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep \
  --adapter /public/home/hh_hh/LoRA/condense/outputs/stage2_gdpo_cite \
  --device cuda:2 \
  --seed 42
```

# LoCoMo reader
## Full Context：
```bash
python experience/scripts/run/run_reader.py \
  --memory-input experience/processed/group1/locomo/compressor_input.jsonl \
  --eval-meta experience/processed/group1/locomo/eval_meta.jsonl \
  --output experience/predictions/full_context/group1/locomo.jsonl \
  --memory-mode full_context \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --device cuda:0 \
  --seed 42
```

## Stage-0
```bash
python experience/scripts/run/run_reader.py \
  --memory-input experience/compressions/stage0/group1/locomo.jsonl \
  --eval-meta experience/processed/group1/locomo/eval_meta.jsonl \
  --output experience/predictions/stage0/group1/locomo.jsonl \
  --memory-mode compressed \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --device cuda:0 \
  --seed 42
```

## Stage-1
```bash
python experience/scripts/run/run_reader.py \
  --memory-input experience/compressions/stage1/group1/locomo.jsonl \
  --eval-meta experience/processed/group1/locomo/eval_meta.jsonl \
  --output experience/predictions/stage1/group1/locomo.jsonl \
  --memory-mode compressed \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --device cuda:0 \
  --seed 42
```
## Stage-2
```bash
python experience/scripts/run/run_reader.py \
  --memory-input experience/compressions/stage2/group1/locomo.jsonl \
  --eval-meta experience/processed/group1/locomo/eval_meta.jsonl \
  --output experience/predictions/stage2/group1/locomo.jsonl \
  --memory-mode compressed \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --device cuda:0 \
  --seed 42
```

=============================================================================


# LongMemoryEval compressor
## Stage-0
```bash
python experience/scripts/run/run_compressor.py \
--input experience/processed/group1/longmemeval/compressor_input.jsonl \
--output experience/compressions/stage0/group1/longmemeval.jsonl \
--base-model /public/home/hh_hh/LoRA/models/Qwen3-4B \
--device cuda:2 \
--seed 42 \
--long-context
```

## Stage-1
```bash
python experience/scripts/run/run_compressor.py \
  --input experience/processed/group1/longmemeval/compressor_input.jsonl \
  --output experience/compressions/stage1/group1/longmemeval.jsonl \
  --base-model /public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep \
  --device cuda:0 \
  --seed 42 \
  --long-context
```
## Stage-2
```bash
python experience/scripts/run/run_compressor.py \
  --input experience/processed/group1/longmemeval/compressor_input.jsonl \
  --output experience/compressions/stage2/group1/longmemeval.jsonl \
  --base-model /public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep \
  --adapter /public/home/hh_hh/LoRA/condense/outputs/stage2_gdpo_cite \
  --device cuda:2 \
  --seed 42 \
  --long-context
```

# LongMemoryEval reader
## Full Context：
```bash
python experience/scripts/run/run_reader.py \
  --memory-input experience/processed/group1/longmemeval/compressor_input.jsonl \
  --eval-meta experience/processed/group1/longmemeval/eval_meta.jsonl \
  --output experience/predictions/full_context/group1/longmemeval.jsonl \
  --memory-mode full_context \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --device cuda:0 \
  --seed 42 \
  --long-context
```

## Stage-0
```bash
python experience/scripts/run/run_reader.py \
  --memory-input experience/compressions/stage0/group1/longmemeval.jsonl \
  --eval-meta experience/processed/group1/longmemeval/eval_meta.jsonl \
  --output experience/predictions/stage0/group1/longmemeval.jsonl \
  --memory-mode compressed \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --device cuda:0 \
  --seed 42
```

## Stage-1
```bash
python experience/scripts/run/run_reader.py \
  --memory-input experience/compressions/stage1/group1/longmemeval.jsonl \
  --eval-meta experience/processed/group1/longmemeval/eval_meta.jsonl \
  --output experience/predictions/stage1/group1/longmemeval.jsonl \
  --memory-mode compressed \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --device cuda:0 \
  --seed 42
```
## Stage-2
```bash
python experience/scripts/run/run_reader.py \
  --memory-input experience/compressions/stage2/group1/longmemeval.jsonl \
  --eval-meta experience/processed/group1/longmemeval/eval_meta.jsonl \
  --output experience/predictions/stage2/group1/longmemeval.jsonl \
  --memory-mode compressed \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --device cuda:0 \
  --seed 42
```

===========================================================================

# GovReport compressor
## Stage-0
```bash
python experience/scripts/run/run_compressor.py \
--input experience/processed/group2/gov_report/compressor_input.jsonl \
--output experience/compressions/stage0/group2/gov_report.jsonl \
--base-model /public/home/hh_hh/LoRA/models/Qwen3-4B \
--device cuda:2 \
--seed 42
```

## Stage-1
```bash
python experience/scripts/run/run_compressor.py \
  --input experience/processed/group2/gov_report/compressor_input.jsonl \
  --output experience/compressions/stage1/group2/gov_report.jsonl \
  --base-model /public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep \
  --device cuda:0 \
  --seed 42
```
## Stage-2
```bash
python experience/scripts/run/run_compressor.py \
  --input experience/processed/group2/gov_report/compressor_input.jsonl \
  --output experience/compressions/stage2/group2/gov_report.jsonl \
  --base-model /public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep \
  --adapter /public/home/hh_hh/LoRA/condense/outputs/stage2_gdpo_cite \
  --device cuda:2 \
  --seed 42
```

# MultiNews compressor
## Stage-0
```bash
python experience/scripts/run/run_compressor.py \
--input experience/processed/group2/multi_news/compressor_input.jsonl \
--output experience/compressions/stage0/group2/multi_news.jsonl \
--base-model /public/home/hh_hh/LoRA/models/Qwen3-4B \
--device cuda:2 \
--seed 42
```

## Stage-1
```bash
python experience/scripts/run/run_compressor.py \
  --input experience/processed/group2/multi_news/compressor_input.jsonl \
  --output experience/compressions/stage1/group2/multi_news.jsonl \
  --base-model /public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep \
  --device cuda:0 \
  --seed 42
```
## Stage-2
```bash
python experience/scripts/run/run_compressor.py \
  --input experience/processed/group2/multi_news/compressor_input.jsonl \
  --output experience/compressions/stage2/group2/multi_news.jsonl \
  --base-model /public/home/hh_hh/LoRA/condense/outputs/sft_full_600_v2_7ep \
  --adapter /public/home/hh_hh/LoRA/condense/outputs/stage2_gdpo_cite \
  --device cuda:2 \
  --seed 42
```


