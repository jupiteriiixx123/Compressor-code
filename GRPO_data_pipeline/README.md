# QuALITY → Stage-2 GRPO Data Pipeline

This package implements the complete offline data-production pipeline:

```text
Step 1  QuALITY parser
        ├─ context
        └─ native probes

Step 2  paragraph-aware context chunking

Step 3  local atomic-information extraction with a frozen large LLM

Step 4  document-level/global relation extraction

Step 5  deterministic merge + exact duplicate removal

Step 6  semantic duplicate detection/canonicalization with a frozen large LLM

Final:
context + native probes + information_units
→ Stage-2 GRPO dataset
```

The compressor must see **only `context`**. `information_units` and `probes` are hidden reward metadata.

## QuALITY handling

The official QuALITY JSONL format contains exactly two rows with the same `article_id` in each split, corresponding to two question writers. `gold_label` is 1-indexed. Step 1 aggregates both rows so one article becomes one GRPO sample with many native probes.

## Recommended server layout

```text
~/LoRA/condense/
├─ benchmarks/
│  └─ quality/
│     ├─ QuALITY.v1.0.1.htmlstripped.train
│     ├─ QuALITY.v1.0.1.htmlstripped.dev
│     └─ QuALITY.v1.0.1.htmlstripped.test
├─ training_data/
│  └─ stage2/
└─ quality_stage2_pipeline/
```

Run commands from the package root.

---

## Step 1 — Parse QuALITY

```bash
python scripts/01_prepare_quality.py \
  --input /public/home/hh_hh/LoRA/condense/benchmarks/quality/QuALITY.v1.0.1.htmlstripped.train \
  --split train \
  --output work/train/articles.jsonl \
  --tokenizer /public/home/hh_hh/LoRA/models/Qwen3-4B
```

Output is article-level:

```json
{
  "id": "quality::12345",
  "context": "...full article...",
  "source_tokens": 6124,
  "probes": [
    {
      "id": "quality::12345::q0001",
      "question": "...",
      "choices": ["...", "...", "...", "..."],
      "answer_index": 2,
      "answer_letter": "C"
    }
  ]
}
```

---

## Step 2 — Chunk contexts

```bash
python scripts/02_chunk_contexts.py \
  --input work/train/articles.jsonl \
  --output work/train/chunks.jsonl \
  --tokenizer /path/to/Qwen-32B-Instruct \
  --chunk-tokens 1600 \
  --overlap-tokens 150
```

Chunking is paragraph-aware; oversized paragraphs are token-sliced. No source text is silently truncated.

---

## Step 3 — Local atomic extraction

Recommended deployment is a local OpenAI-compatible endpoint, e.g. vLLM:

```bash
vllm serve /path/to/Qwen-32B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --port 8000
```

Then:

```bash
python scripts/03_extract_local_units.py \
  --input work/train/chunks.jsonl \
  --output work/train/local_units.jsonl \
  --backend openai_compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen-32B-Instruct \
  --workers 4 \
  --max-new-tokens 4096
```

For a pilot, add `--limit 10 --keep-raw`.

The step is resumable: already-written record IDs are skipped. Errors are written to `*.errors.jsonl`.

Direct HF loading is also supported:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/03_extract_local_units.py \
  --input work/train/chunks.jsonl \
  --output work/train/local_units.jsonl \
  --backend hf \
  --model /path/to/Qwen-32B-Instruct \
  --workers 1 \
  --attn-implementation sdpa \
  --max-model-tokens 32768 \
  --max-new-tokens 4096
```

The HF backend uses `device_map="auto"` and refuses silent overflow truncation.

---

## Step 4 — Global relation extraction

```bash
python scripts/04_extract_global_units.py \
  --articles work/train/articles.jsonl \
  --chunks work/train/chunks.jsonl \
  --local-units work/train/local_units.jsonl \
  --output work/train/global_units.jsonl \
  --backend openai_compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen-32B-Instruct \
  --workers 4 \
  --max-new-tokens 4096
```

This pass receives the full document plus the local claims and is instructed to add only missing long-range relations.

---

## Step 5 — Deterministic merge

```bash
python scripts/05_merge_units.py \
  --articles work/train/articles.jsonl \
  --local-units work/train/local_units.jsonl \
  --global-units work/train/global_units.jsonl \
  --output work/train/merged_units.jsonl
```

This step performs only deterministic operations: merge, stable IDs, exact normalized duplicate removal, and provenance merge. It never rewrites semantic content.

---

## Step 6 — Semantic deduplication

```bash
python scripts/06_semantic_dedup.py \
  --input work/train/merged_units.jsonl \
  --output work/train/final_units.jsonl \
  --backend openai_compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model /path/to/Qwen-32B-Instruct \
  --workers 4 \
  --max-new-tokens 4096
```

The dedup model returns only duplicate ID groups, for example:

```json
{
  "duplicate_groups": [
    {
      "representative_id": "u00003",
      "member_ids": ["u00003", "u00047"]
    }
  ]
}
```

The final canonical claim is copied from the representative source unit rather than regenerated. Partial overlap, entailment, or a more-specific/less-specific relation is explicitly not treated as duplication.

---

## Build final Stage-2 data

```bash
python scripts/07_build_stage2_dataset.py \
  --articles work/train/articles.jsonl \
  --final-units work/train/final_units.jsonl \
  --split train \
  --output-dir training_data/stage2/quality
```

Outputs:

```text
stage2_train.jsonl   # unified audit file
train_policy.jsonl   # compressor-visible data only
train_reward.jsonl   # hidden reward metadata only
```

`train_policy.jsonl`:

```json
{"id":"quality::12345","context":"...","source_tokens":6124}
```

`train_reward.jsonl`:

```json
{
  "id": "quality::12345",
  "information_units": [
    {
      "id": "u00001",
      "claim": "...",
      "type": "event",
      "provenance": []
    }
  ],
  "probes": [
    {
      "id": "quality::12345::q0001",
      "question": "...",
      "choices": ["...", "...", "...", "..."],
      "answer_letter": "C"
    }
  ]
}
```

The physical split makes accidental QA leakage into the compressor harder.

---

## Validate before GRPO

```bash
python scripts/08_validate_stage2_dataset.py \
  --input training_data/stage2/quality/stage2_train.jsonl \
  --report training_data/stage2/quality/train_report.json
```

The report includes source-token statistics, probes/article, units/article, units per 1K source tokens, missing answers, duplicate IDs, empty claims, and local evidence exact-match rate.

Before scaling, manually inspect roughly 10 articles / 200–300 units. The most important failure mode is missing source information.

---

## Switching to a commercial API

No pipeline code changes are needed if the provider offers an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="..."
python scripts/03_extract_local_units.py \
  ... \
  --backend openai_compatible \
  --base-url https://provider.example/v1 \
  --model provider-model-name \
  --api-key-env OPENAI_API_KEY
```

If required:

```bash
--max-token-field max_completion_tokens
```

If supported:

```bash
--json-mode
```

---

## Design invariants

1. Compressor input is `context` only.
2. `probes` and `information_units` are reward metadata.
3. Extraction is exhaustive, not importance-ranked summarization.
4. Local extraction is recall-oriented.
5. Global extraction repairs missing long-range relations.
6. Deterministic merge does not make semantic decisions.
7. Semantic dedup merges only equivalent claims.
8. Evidence/provenance is retained for audit.
9. Direct HF inference performs no silent input truncation.
10. QuALITY dev is processed identically but used only for validation.
