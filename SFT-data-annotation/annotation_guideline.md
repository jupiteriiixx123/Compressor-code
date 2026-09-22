# Long-Context Compression Annotation Guideline

## 1. Task

Given one long document, produce a structured compressed representation of the document.

The representation contains:

1. `summary`: a compact, coherent backbone of the whole document.
2. `link`: detailed information blocks referenced from the summary.
3. `evidence`: paragraph IDs from the source document that support each information block.

The goal is NOT ordinary summarization.

The goal is to preserve as much useful information as possible in a compact and structured representation, so that a downstream language model can answer future questions about the original document without reading the full document.

The future questions are unknown during compression.

Therefore, annotation must be query-agnostic.


## 2. Output format

Output exactly one valid JSON object:

{
  "id": "original_document_id",

  "summary": "...[cite_1]...",

  "link": {
    "cite_1": "...",
    "cite_2": "..."
  },

  "evidence": {
    "cite_1": ["P0001", "P0015"],
    "cite_2": ["P0032", "P0067"]
  }
}

Do not output Markdown fences.
Do not output explanations before or after the JSON.


## 3. Citation format

Citation markers MUST use ASCII English square brackets:

[cite_1]
[cite_2]
[cite_3]

Do NOT use:

【cite_1】
(cite_1)
[CITE1]

Citation IDs must start from `cite_1` and increase consecutively.


## 4. Summary

The `summary` is the backbone of the document.

It should allow a reader to understand the document's:

- overall subject
- main purpose or problem
- major entities
- major events or developments
- key findings
- important relationships
- causal relationships
- major comparisons
- major conclusions
- important recommendations or implications

The summary must be coherent and readable by itself.

The summary should NOT attempt to preserve every detail.

Detailed information that is useful but would make the summary too dense should be moved into information blocks and referenced using `[cite_n]`.


## 5. Adaptive compression

The size and structure of the compressed representation MUST be determined independently for each document according to its actual information content.

There is NO preferred, typical, default, minimum, or target value for:

- summary length
- number of citations
- number of information blocks
- information-block length
- total compressed length
- compression ratio

Do NOT aim for consistency in these quantities across documents.

Do NOT imitate the length, citation count, or structure of previously annotated documents.

A document may naturally require very few information blocks, while another document may require many more.

For example, 2 citations, 6 citations, 12 citations, or substantially more may all be appropriate depending on the document. These numbers are examples only and MUST NOT be treated as targets or preferred ranges.

Similarly, a short summary may be appropriate for one document while a substantially longer summary may be necessary for another.

The only criterion is whether the representation preserves the document's important information efficiently and faithfully.

Do not add citations merely to increase coverage mechanically.
Do not merge distinct important information merely to reduce the citation count.

Structure is fixed; information quantity is adaptive.


## 6. Citation placement

A citation marker `[cite_n]` is a semantic anchor to an expandable information block.

Place a citation immediately after the smallest meaningful span of text that the corresponding information block elaborates.

The anchor may be:

- a technical term
- a concept
- an entity
- a method or model name
- a phrase
- a clause
- a factual statement
- or a complete sentence


Examples:

The method introduces dynamicity[cite_1] as a measure of how rapidly a community's interests change over time.

The model combines a BiGRU encoder[cite_2] with discourse-based polarity propagation[cite_3].

The proposed method substantially improves low-resource performance, although noisy causal relations remain an important limitation.[cite_4]

A single sentence may contain:
- no citation,
- one citation,
- or multiple citations,

depending on how many independently useful details can be expanded.

Likewise, not every summary sentence needs an information block.

Create a citation only when the referenced block provides useful additional information beyond what is already present in the summary.

The citation must have a clear semantic relationship with the exact term, phrase, clause, or statement that precedes it.

Do not place citations arbitrarily.

Do not force a "one sentence = one citation = one information block" structure.


Citation structure must emerge from information needs, not from sentence structure or document sections.

Do NOT create one citation mechanically for every summary sentence.

Do NOT create one information block for every major section, such as
Introduction, Method, Experiments, Results, and Limitations.

A summary sentence may:
- require no citation,
- use one citation,
- or, when genuinely useful, refer to multiple information blocks.

Likewise, one information block may synthesize details from multiple sections or distant parts of the document.

Create an information block only when there is useful supporting or recoverable detail that should remain accessible beyond what is already stated in the summary.

The summary should first be written as a coherent semantic backbone.
Citation boundaries should then follow the natural structure of the information,
rather than determining the structure of the summary.


## 7. Information blocks

Each entry in `link` is a compact semantic information block.

An information block should:

- focus on one coherent topic
- preserve important factual details omitted from the summary
- remove irrelevant wording
- merge repetitive information
- reorganize information when this improves clarity
- preserve important numbers, dates, names, conditions, exceptions, and relationships when relevant
- combine related information that may be distributed across different parts of the document

An information block is NOT required to correspond to one contiguous source paragraph.

For example, information about the same policy, entity, experiment, event, or conclusion may appear in paragraphs P0012, P0041, and P0087.

If these pieces are semantically related, they may be consolidated into one block.


## 8. Cross-paragraph consolidation

Cross-paragraph consolidation is encouraged when it is semantically justified.

Example:

P0010 describes the policy.
P0035 describes an implementation problem.
P0072 describes the eventual outcome.

If these three passages jointly describe one coherent subject, they may be consolidated into one information block.

However, do NOT combine unrelated information simply because it appears important.

Semantic coherence is more important than the number of source paragraphs.


## 9. Do not copy raw text unnecessarily

Information blocks should normally be synthesized rather than copied verbatim from the source.

Do not simply paste a full paragraph into `link`.

Rewrite and consolidate the source into a denser form while preserving factual meaning.

Short exact terminology, names, numerical values, or phrases may naturally remain unchanged when necessary.


## 10. Evidence

For every citation in `link`, provide an entry with the same key in `evidence`.

Example:

"link": {
  "cite_4": "..."
},

"evidence": {
  "cite_4": ["P0012", "P0048", "P0071"]
}

Evidence IDs must refer only to paragraph IDs that actually exist in the input document.

Evidence should identify the source paragraphs that directly support the information block.

Evidence should be minimal but sufficient. Every listed evidence paragraph must materially support at least one fact contained in the corresponding information block.

Do not include a continuous range of paragraphs merely because they belong to the same section or discuss the same general topic.

When a block consolidates information from multiple locations, include all and only the source paragraphs necessary to support the synthesized information.

A paragraph should not be included if removing it would leave all factual claims in the information block equally well supported.


## 11. Faithfulness

Do not introduce facts that are not supported by the source document.

Do not:

- infer unsupported causal relationships
- invent numbers
- invent dates
- invent conclusions
- strengthen uncertain claims into definite claims
- remove important qualifications or exceptions

Compression may reorganize information, but it must preserve the meaning of the source.


## 12. Information selection

When deciding what to preserve, prioritize information that may plausibly matter for future questions about the document.

Important examples include:

- major claims and findings
- key evidence
- important entities and their roles
- quantitative results
- methods or procedures when important to understanding the document
- important comparisons
- causes and consequences
- timelines
- conditions and exceptions
- disagreements
- limitations
- recommendations
- final conclusions

Low-value repetition, boilerplate, procedural filler, contact information, and other material with little semantic value can usually be omitted.


## 13. Document-specific judgment

Different document types require different compression strategies.

For government reports:
prioritize the investigated problem, findings, evidence, policy details, failures, comparisons, recommendations, agency responses, and important quantitative results.

For scientific papers:
prioritize the research problem, motivation, method, assumptions, experimental setup when necessary, major results, comparisons, ablations, limitations, and conclusions.

Do not mechanically apply the same structure to every document.


## 14. Internal consistency requirements

Before saving the output, verify:

1. The output is valid JSON.
2. `id` exactly matches the input document ID.
3. Every citation appearing in `summary` exists in `link`.
4. Every citation appearing in `summary` exists in `evidence`.
5. Every key in `link` appears in `summary`.
6. Every key in `evidence` also exists in `link`.
7. Citation numbering is consecutive.
8. Every evidence paragraph ID exists in the input document.
9. Information blocks are faithful to their evidence.
10. The summary remains understandable without opening any information block.


## 15. Core principle

A good annotation should maximize:

information retained / text required

while remaining faithful, coherent, and useful for answering unknown future questions about the original document.