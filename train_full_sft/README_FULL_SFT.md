# Qwen3-4B Full SFT (3 × L20)

Recommended first run:

```bash
python - <<'PY'
import deepspeed, transformers, trl
from importlib.metadata import version
print('deepspeed   =', deepspeed.__version__)
print('transformers=', transformers.__version__)
print('trl         =', trl.__version__)
print('liger       =', version('liger-kernel'))
PY
```

Expected Liger version in the working environment: `0.5.10`.

Launch the speed-first ZeRO-3 run:

```bash
CUDA_VISIBLE_DEVICES=0,1,4 torchrun --standalone --nproc_per_node=3   scripts/train_sft_full.py   --data training_data/sft_train.jsonl   --model /public/home/hh_hh/LoRA/models/Qwen3-4B   --output-dir outputs/sft_full_600_v1   --deepspeed configs/ds_zero3_full.json   --max-length 40960   --num-train-epochs 3   --learning-rate 2e-5   --per-device-train-batch-size 1   --gradient-accumulation-steps 3   --attn-implementation sdpa   --save-steps 50   --logging-steps 1
```

This matches the LoRA run on dataset, max length, epochs and effective batch
size (= 3 GPUs × 1 × 3 = 9). The intentional differences are full-parameter
updates, DeepSpeed ZeRO-3, and a lower learning rate (2e-5).

If GPU OOM occurs because model/optimizer state leaves too little activation
headroom, rerun with `configs/ds_zero3_full_offload_optimizer.json`. It is
slower but frees substantial GPU memory by moving optimizer state to CPU.

The script excludes samples above 40960 tokens without truncation, so with the
current dataset it should keep the same 598 examples used by the LoRA run.



# Qwen3-4B Full SFT (4 × L20)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_full_sft/train_sft_full.py \
  --data training_full_sft/training_data/sft_train.jsonl \
  --model /public/home/hh_hh/LoRA/models/Qwen3-4B \
  --output-dir outputs/sft_full_600_v2_5p \
  --deepspeed train_full_sft/ds_zero3_full.json \
  --max-length 40960 \
  --num-train-epochs 5 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --attn-implementation sdpa \
  --save-steps 50 \
  --logging-steps 1
```

python scripts/01_prepare_quality.py \
  --input /public/home/hh_hh/LoRA/condense/benchmarks/quality/data/QuALITY.v1.0.1.htmlstripped.dev \
  --split dev \
  --output work/dev/articles.jsonl \
  --tokenizer /public/home/hh_hh/LoRA/models/Qwen3-4B

python scripts/02_chunk_contexts.py \
  --input work/train/articles.jsonl \
  --output work/train/chunks.jsonl \
  --tokenizer /public/home/hh_hh/LoRA/models/Qwen3.8-27B \
  --chunk-tokens 1600 \
  --overlap-tokens 150


