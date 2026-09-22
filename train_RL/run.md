```bash
CUDA_VISIBLE_DEVICES=1,2 \
python train_RL/train.py \
  --policy-data GRPO_data_pipeline/work/train/final/train_policy.jsonl \
  --reward-data GRPO_data_pipeline/work/train/final/train_reward.jsonl \
  --stage1-model outputs/sft_full_600_v2_7ep \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --output-dir outputs/stage2_gdpo \
  --advantage gdpo \
  --group-size 4 \
  --batch-size 1 \
  --epochs 1 \
  --update-epochs 2 \
  --save-every 10 \
  --policy-device cuda:0 \
  --reader-device cuda:1
```

中断后继续
```bash
CUDA_VISIBLE_DEVICES=1,2 \
python train_RL/train.py \
  --policy-data GRPO_data_pipeline/work/train/final/train_policy.jsonl \
  --reward-data GRPO_data_pipeline/work/train/final/train_reward.jsonl \
  --stage1-model outputs/sft_full_600_v2_7ep \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --output-dir outputs/stage2_gdpo_cite \
  --advantage gdpo \
  --group-size 4 \
  --batch-size 1 \
  --epochs 2 \
  --update-epochs 2 \
  --save-every 30 \
  --resume-from outputs/stage2_gdpo_cite/checkpoint-180 \
  --policy-device cuda:0 \
  --reader-device cuda:1
```

debug训练
```bash
CUDA_VISIBLE_DEVICES=1,2 \
python train_RL/train.py \
  --policy-data GRPO_data_pipeline/work/train/final/train_policy.jsonl \
  --reward-data GRPO_data_pipeline/work/train/final/train_reward.jsonl \
  --stage1-model outputs/sft_full_600_v2_7ep \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --output-dir outputs/stage2_gdpo-cite \
  --advantage gdpo \
  --group-size 4 \
  --batch-size 1 \
  --epochs 1 \
  --update-epochs 2 \
  --debug \
  --max-steps 3 \
  --monitor-size 0 \
  --policy-device cuda:0 \
  --reader-device cuda:1
```

正式训练
```bash
CUDA_VISIBLE_DEVICES=1,2 \
python train_RL/train.py \
  --policy-data GRPO_data_pipeline/work/train/final/train_policy.jsonl \
  --reward-data GRPO_data_pipeline/work/train/final/train_reward.jsonl \
  --stage1-model outputs/sft_full_600_v2_7ep \
  --reader-model /public/home/hh_hh/LoRA/models/Qwen3-8B \
  --output-dir outputs/stage2_gdpo_long \
  --advantage gdpo \
  --group-size 4 \
  --batch-size 1 \
  --epochs 2 \
  --update-epochs 2 \
  --reward-weights 1 2 6 0.2 \
  --target-ratio 0.30 \
  --reader-batch-size 16 \
  --monitor-size 3 \
  --monitor-every 30 \
  --save-every 30 \
  --debug \
  --policy-device cuda:0 \
  --reader-device cuda:1
```