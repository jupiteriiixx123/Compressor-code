
```bash
python scripts/01_prepare_quality.py \
  --input /public/home/hh_hh/LoRA/condense/benchmarks/quality/data/QuALITY.v1.0.1.htmlstripped.dev \
  --split dev \
  --output work/dev/articles.jsonl \
  --tokenizer /public/home/hh_hh/LoRA/models/Qwen3-4B
```

```bash
python scripts/02_chunk_contexts.py \
  --input work/dev/articles.jsonl \
  --output work/dev/chunks.jsonl \
  --tokenizer /public/home/hh_hh/LoRA/models/Qwen3.8-27B \
  --chunk-tokens 1600 \
  --overlap-tokens 150
```
GLM推理服务
```bash
CUDA_VISIBLE_DEVICES=1,2 \
vllm serve /public/home/hh_hh/LoRA/models/GLM-4-32B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```
```bash
python scripts/03_extract_local_units.py \
  --input work/train/chunks.jsonl \
  --output work/train/local_units_glm_pilot_v2.jsonl \
  --backend openai_compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model /public/home/hh_hh/LoRA/models/GLM-4-32B \
  --workers 2 \
  --max-new-tokens 8192 \
  --limit 10 \
  --keep-raw
```

Qwen推理服务
```bash
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8001
```
```bash
cd ~/LoRA/condense/GRPO_data_pipeline

python scripts/03_extract_local_units.py \
  --input work/train/chunks.jsonl \
  --output work/train/local_units_qwen.jsonl \
  --backend openai_compatible \
  --base-url http://127.0.0.1:8001/v1 \
  --model /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --workers 2 \
  --max-new-tokens 8192
```


hf download Qwen/Qwen2.5-32B-Instruct \
  --local-dir /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct

正式GLM提取atomic units
```bash
python scripts/03_extract_local_units.py \
  --input work/train/chunks.jsonl \
  --output work/train/local_units_glm.jsonl \
  --backend openai_compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model /public/home/hh_hh/LoRA/models/GLM-4-32B \
  --workers 2 \
  --max-new-tokens 8192
```

```bash
CUDA_VISIBLE_DEVICES=3,4 \
vllm serve /public/home/hh_hh/LoRA/models/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8001
```


final data 合成
```bash
cd ~/LoRA/condense/GRPO_data_pipeline

python scripts/07_build_stage2_dataset.py \
  --articles work/train/articles.jsonl \
  --local-units work/train/local_units_glm.jsonl \
  --split train \
  --output-dir work/train/final
```