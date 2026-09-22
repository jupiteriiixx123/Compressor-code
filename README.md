# Index-Compressor

Long-Context Structured Compression with SFT + RL.

Index-Compressor 核心训练代码与数据构造代码

面向 LLM 长上下文管理场景，通过结构化压缩方法保留关键信息


## Project Structure

SFT-data-annotation       --- SFT 训练数据构造与标注流程

train_full_sft           --- Full Parameter SFT 训练代码

train_lora_sft           --- LoRA-based SFT 训练代码

train_RL                  --- RL post-training 训练流程

GRPO_data_pipeline      --- GRPO 数据处理与训练 pipeline

scripts                 --- 实验运行脚本与辅助工具
