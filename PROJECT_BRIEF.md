# MergeRAG Project Brief

## Project Identity

**Name:** MergeRAG  
**Subtitle:** Query-Aware Context Merging  
**Reference paper:** arXiv 2603.20286, "Rethinking Retrieval-Augmentation as Synthesis: A Query-Aware Context Merging Approach"

MergeRAG is a flagship portfolio and research-system project. The goal is to build a deployable, benchmark-first implementation that demonstrates whether query-aware context merging improves retrieval-augmented generation over standard Top-k RAG under controlled, visible conditions.

## Core Idea

Traditional RAG retrieves chunks, ranks them, keeps the top `k`, and discards the rest. MergeRAG treats retrieval context as something to synthesize. It retrieves more evidence than it can directly use, then merges weak or redundant evidence into a compact, query-aware context before final answer generation.

The system should support:

- **Top-k RAG:** baseline retrieve-rerank-answer flow.
- **Symmetric merging:** combine weak low-ranked chunks to recover bridge evidence.
- **Asymmetric merging:** fold redundant weak chunks into stronger anchor chunks.
- **Hierarchical parallel merging:** merge independent pairs in batches to reduce latency and repeated rewriting.

Every merged context must preserve provenance back to original chunks.

## Product Direction

The first complete version should be a **1-2 month flagship build** aimed at recruiters and technical reviewers. It should communicate: "I can understand a recent paper, reproduce its core mechanisms, evaluate it rigorously, and deploy it as a usable research product."

Primary deliverables:

- Polished GitHub repository.
- Live deployed demo.
- Benchmark report explaining methodology, findings, limitations, and failure cases.

The app should be minimal but credible: a clean comparison interface, not a flashy landing page. The benchmark and traceability engine are more important than heavy UI effects.

## Technical Direction

Default architecture:

- **Backend:** FastAPI.
- **Frontend:** Next.js.
- **Jobs:** worker queue for ingestion and benchmark runs.
- **Deployment:** Docker Compose on a VPS.
- **Vector store:** Qdrant.
- **Embeddings:** pluggable, with local CPU-friendly embeddings as the default and hosted embeddings optional.
- **LLMs:** multi-provider adapter layer for hosted LLMs.
- **Storage:** relational database for documents, chunks, runs, metrics, prompts, traces, costs, and artifacts.

Public deployment behavior:

- Curated demos and reports are visible publicly.
- Expensive live runs require the viewer to provide their own API key.
- Cached curated runs should be available without API keys.

## Benchmark Direction

First-class datasets:

- HotpotQA.
- MuSiQue.

Initial benchmark scope:

- Small public subsets for fast, reproducible iteration.
- Tiny committed fixtures for tests.
- Download scripts for benchmark subsets.

Required comparisons:

- Top-k baseline.
- Symmetric merge.
- Asymmetric merge.
- Combined/hierarchical MergeRAG.
- Multiple context token budgets.

Metrics:

- Exact Match and F1 where labels exist.
- Context token count.
- Latency.
- API/model cost.
- Citation coverage.
- Merge trace completeness.
- Failure cases and qualitative analysis.

## Reproducibility Requirements

Runs should be explainable after the fact. Store and expose:

- Query.
- Dataset/corpus.
- Retrieval and reranking config.
- Merge mode.
- Token budget.
- Prompt version.
- Model/provider version.
- Retrieved chunks and scores.
- Merge plan and intermediate merged contexts.
- Final answer.
- Citations to source chunks.
- Evaluation metrics.
- Cost and latency metadata.

Prompts should be visible and versioned.

## Citation and Provenance Policy

Citation behavior should be strict. Every final answer sentence should map to source chunk IDs where possible. Merged chunks must retain lineage to the original document chunks that contributed evidence.

The system should avoid unsupported synthesis during merging. Merge prompts should be extraction-heavy and should preserve source wording where practical.

## UI Requirements

The showcase app should let viewers:

- Select curated research-paper corpora.
- Upload documents in local/private use.
- Ask one query across Top-k RAG and MergeRAG modes.
- Compare answers side by side.
- Inspect retrieved chunks, scores, merge steps, citations, token usage, costs, and latency.
- View benchmark reports and failure cases.

The UI should feel like a research dashboard: clean, legible, inspectable, and professional.

## CI and Testing Direction

CI should run core checks only:

- Unit tests.
- Golden tests using tiny fixtures.
- Type/lint checks once the stack exists.

Full benchmark runs should remain manual or scheduled outside normal CI because they may require model keys, larger caches, and longer runtime.

## Important Assumptions

- The project should implement algorithm and evaluation parity with the paper as closely as practical.
- Exact paper-level reproduction is desirable, but model, prompt, or infrastructure differences must be documented when unavoidable.
- Claims should be measured, not overstated.
- The app/report should visibly include failure cases.
- Correctness, traceability, and reproducibility are more important than raw speed in the first complete version.
