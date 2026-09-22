```bash
CUDA_VISIBLE_DEVICES=1,3 \
python experience/scripts/eval/evaluate_group1.py \
  --dataset locomo \
  --eval-meta experience/processed/group1/locomo/eval_meta.jsonl \
  --prediction full_context=experience/predictions/full_context/group1/locomo.jsonl \
  --prediction stage0=experience/predictions/stage0/group1/locomo.jsonl \
  --prediction stage1=experience/predictions/stage1/group1/locomo.jsonl \
  --prediction stage2=experience/predictions/stage2/group1/locomo.jsonl \
  --output-dir experience/results/group1/locomo_qwen32B \
  --judge-model /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --judge-device-map auto \
  --judge-dtype bf16 \
  --judge-attn-implementation sdpa \
  --judge-batch-size 4
```

提取原子信息
```bash
CUDA_VISIBLE_DEVICES=0,4 \
python experience/scripts/eval/prepare_atomic_cache.py \
  --dataset gov_report \
  --eval-meta experience/processed/group2/gov_report/eval_meta.jsonl \
  --output experience/cache/gov_report_gold_units.jsonl \
  --backend hf \
  --model /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --pipeline-root /public/home/hh_hh/LoRA/condense/GRPO_data_pipeline \
  --attn-implementation sdpa \
  --max-model-tokens 32768 \
  --max-new-tokens 4096 \
  --workers 1
```

```bash
CUDA_VISIBLE_DEVICES=1,3 \
python experience/scripts/eval/prepare_atomic_cache.py \
  --dataset multi_news \
  --eval-meta experience/processed/group2/multi_news/eval_meta.jsonl \
  --output experience/cache/multi_news_gold_units.jsonl \
  --backend hf \
  --model /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --pipeline-root /public/home/hh_hh/LoRA/condense/GRPO_data_pipeline \
  --attn-implementation sdpa \
  --max-model-tokens 32768 \
  --max-new-tokens 4096 \
  --workers 1
```

GovReport:
```bash
CUDA_VISIBLE_DEVICES=0,4 \
python experience/scripts/eval/evaluate_group2.py \
  --dataset gov_report \
  --compressor-input experience/processed/group2/gov_report/compressor_input.jsonl \
  --eval-meta experience/processed/group2/gov_report/eval_meta.jsonl \
  --gold-units experience/cache/gov_report_gold_units.jsonl \
  --compression stage2=experience/compressions/stage2/group2/gov_report.jsonl \
  --output-dir experience/results/group2/gov_report_stage2_prelim \
  --judge-model /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --judge-device-map auto \
  --judge-dtype bf16 \
  --judge-attn-implementation sdpa \
  --judge-batch-size 1 \
  --ratio-tokenizer /public/home/hh_hh/LoRA/models/Qwen3-4B \
  --allow-missing \
  --max-samples 5
```

```bash
CUDA_VISIBLE_DEVICES=0,4 \
python experience/scripts/eval/evaluate_group2.py \
  --dataset gov_report \
  --compressor-input experience/processed/group2/gov_report/compressor_input.jsonl \
  --eval-meta experience/processed/group2/gov_report/eval_meta.jsonl \
  --gold-units experience/cache/gov_report_gold_units.jsonl \
  --compression stage1=experience/compressions/stage1/group2/gov_report.jsonl \
  --output-dir experience/results/group2/gov_report_stage1_main \
  --judge-model /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --judge-device-map auto \
  --judge-dtype bf16 \
  --judge-attn-implementation sdpa \
  --judge-batch-size 1 \
  --ratio-tokenizer /public/home/hh_hh/LoRA/models/Qwen3-4B \
  --allow-missing
```

```bash
CUDA_VISIBLE_DEVICES=1,3 \
python experience/scripts/eval/evaluate_group2.py \
  --dataset gov_report \
  --compressor-input experience/processed/group2/gov_report/compressor_input.jsonl \
  --eval-meta experience/processed/group2/gov_report/eval_meta.jsonl \
  --gold-units experience/cache/gov_report_gold_units.jsonl \
  --compression stage0=experience/compressions/stage0/group2/gov_report.jsonl \
  --output-dir experience/results/group2/gov_report_stage0_main \
  --judge-model /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --judge-device-map auto \
  --judge-dtype bf16 \
  --judge-attn-implementation sdpa \
  --judge-batch-size 1 \
  --ratio-tokenizer /public/home/hh_hh/LoRA/models/Qwen3-4B \
  --allow-missing
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python experience/scripts/eval/evaluate_group2_full_only.py \
  --dataset gov_report \
  --compressor-input experience/processed/group2/gov_report/compressor_input.jsonl \
  --eval-meta experience/processed/group2/gov_report/eval_meta.jsonl \
  --gold-units experience/cache/gov_report_gold_units.jsonl \
  --output-dir experience/results/group2/gov_report_full_context \
  --judge-model /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --ratio-tokenizer /public/home/hh_hh/LoRA/models/Qwen3-4B \
  --allow-missing
```