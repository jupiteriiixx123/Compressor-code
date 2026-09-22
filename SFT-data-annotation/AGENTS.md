# Annotation Batch Task

Before annotating documents, read and follow strictly:

`annotation_guideline.md`

## Input

Recursively scan all `.json` files under:

`annotation_inputs/`

## Output

For every input file, write the annotation under:

`annotation_outputs/`

Preserve the same relative subdirectory and filename.

Example:

Input:
`annotation_inputs/crs/govreport_crs_RL33318.json`

Output:
`annotation_outputs/crs/govreport_crs_RL33318.json`

## Resume / checkpoint behavior

Before annotating an input document, check whether its expected output file already exists.

If the output file:

1. exists,
2. is valid JSON, and
3. its `id` exactly matches the input document's `id`,

then treat that document as completed and skip it.

Do not regenerate or overwrite a valid completed annotation.

If the output file does not exist, annotate the document normally.

If the output file exists but is invalid JSON, incomplete, or has a mismatched `id`, regenerate that document and replace the invalid output.

This rule is the checkpoint/resume mechanism.

## Annotation generation requirement

Every annotation MUST be produced by your own language-model reasoning after reading the corresponding input document.

You must personally read and semantically compress each document.

Do NOT delegate annotation generation to:

- a local language model
- another model
- transformers
- llama.cpp
- MLX
- external APIs
- heuristic summarizers
- extractive summarization algorithms
- deterministic compression scripts
- keyword-based rules
- section-selection rules
- templates that programmatically generate summary or link content

Scripts may only be used for:
- discovering input files
- checking completed output files
- reading/writing JSON
- validating output structure
- counting progress

Scripts MUST NOT generate, rewrite, select, summarize, or compress document content.

The semantic content of `summary` and `link` must be generated directly by you from the full input document according to `annotation_guideline.md`.

If the number of remaining documents is too large to complete in one run, process as many documents as practical and stop cleanly. Do NOT replace model-based annotation with an automated approximation.

## Independent annotation requirement

Treat every document as an independent annotation task.

Do not infer a preferred summary length, citation count, information-block count, or compression ratio from previously processed documents.

Existing annotation outputs may only be checked to determine whether a document has already been completed.

Do NOT read completed annotation outputs for stylistic guidance or use them as examples for subsequent documents.

Determine the appropriate compression structure solely from the current input document and `annotation_guideline.md`.

## Safety

Never modify files under `annotation_inputs/`.

Never read or use files under `sidecar/`.

Only use the input document itself and `annotation_guideline.md`
for generating an annotation.