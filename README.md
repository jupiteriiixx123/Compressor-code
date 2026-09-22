# Index-Compressor

Long-Context Structured Compression with SFT + RL.

本项目为 Index-Compressor 的核心训练代码与数据构造代码，
面向 LLM 长上下文管理场景，通过结构化压缩方法保留关键信息，
并结合 SFT 与 RL post-training 优化压缩效果。

## Project Structure

.
├── SFT-data-annotation      # SFT 训练数据构造与标注流程
├── train_full_sft           # Full Parameter SFT 训练代码
├── train_lora_sft           # LoRA-based SFT 训练代码
├── train_RL                # RL post-training 训练流程
├── GRPO_data_pipeline      # GRPO 数据处理与训练 pipeline
└── scripts                 # 实验运行脚本与辅助工具


## Training Pipeline

1. Data Construction  
   - SFT 数据生成与标注

2. Supervised Fine-tuning  
   - Full SFT
   - LoRA SFT

3. Reinforcement Learning  
   - GRPO-based post-training

## Note

本仓库包含论文项目中的核心实现代码，
部分训练数据、模型权重及实验配置未公开。
