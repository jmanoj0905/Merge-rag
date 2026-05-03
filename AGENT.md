# AGENT.md

## Mission

Build MergeRAG as a benchmark-first, deployable research product that reproduces the core ideas of arXiv 2603.20286 and showcases them clearly.

This is not just another RAG demo. Treat it as a paper reproduction plus productized research dashboard.

## Current Project State

This repository starts as a fresh greenfield project. The authoritative project intent is captured in `PROJECT_BRIEF.md`.

Before implementing, read:

1. `PROJECT_BRIEF.md`
2. The MergeRAG paper: arXiv 2603.20286
3. Any future architecture or task documents added to this repo

## Engineering Priorities

Prioritize in this order:

1. Correctness and provenance.
2. Reproducibility.
3. Benchmark credibility.
4. Clear deployed showcase.
5. Runtime performance.

Do not optimize away traceability. Every retrieval, merge, and answer step should be inspectable.

## Expected Architecture

Default stack:

- FastAPI backend.
- Next.js frontend.
- Worker queue for ingestion and benchmark jobs.
- Qdrant vector store.
- Relational database for metadata, runs, prompts, metrics, traces, and artifacts.
- Docker Compose deployment.
- Pluggable embedding providers.
- Multi-provider hosted LLM adapter.

Keep provider-specific logic behind adapters. Do not scatter OpenAI, Anthropic, Gemini, or local-model assumptions through core pipeline code.

## Core Behaviors

The system must support these modes:

- `top_k`: standard RAG baseline.
- `symmetric_merge`: merge weak low-ranked chunks to recover bridge evidence.
- `asymmetric_merge`: fold weaker redundant chunks into stronger anchors.
- `combined` or `hierarchical_merge`: use parallel layered merging to fit a token budget.

Merged chunks must retain source lineage to original chunk IDs.

## Benchmark Policy

First-class benchmark targets:

- HotpotQA.
- MuSiQue.

Use tiny committed fixtures for tests and scripts for downloading public benchmark subsets. Do not commit large benchmark datasets or generated caches.

Mandatory report comparisons:

- Top-k baseline.
- Symmetric merge.
- Asymmetric merge.
- Combined/hierarchical MergeRAG.
- Multiple context token budgets.

## Prompt and Trace Policy

Prompts are part of the reproducibility surface.

Store prompt versions and expose them in reports or debug views. Each run should capture:

- Query.
- Corpus/dataset.
- Retrieval config.
- Merge config.
- Model/provider config.
- Prompt versions.
- Retrieved chunks and scores.
- Merge plan.
- Intermediate merged outputs.
- Final answer.
- Citations.
- Token usage.
- Cost.
- Latency.
- Evaluation metrics when available.

## Citation Policy

Use strict citations. Every final answer sentence should map to supporting source chunks where possible.

Avoid unsupported generated facts during merging. Merge operations should be extraction-heavy, query-aware, and provenance-preserving.

## Public Demo Policy

The deployed app should be safe to share publicly:

- Public viewers can inspect curated demos, cached runs, and reports.
- Live model calls require viewer-provided API keys unless explicitly configured otherwise.
- Expensive benchmark runs should not be publicly triggerable without protection.

## UI Direction

The frontend should be a minimal, polished research dashboard.

Required user-facing capabilities:

- Select curated corpora.
- Compare Top-k RAG and MergeRAG side by side.
- Inspect retrieved chunks, scores, merge traces, citations, token usage, cost, and latency.
- View benchmark reports and failure cases.

Avoid overbuilding marketing pages before the core comparison experience exists.

## Testing Expectations

Add tests as functionality lands.

Important test categories:

- Chunking and ingestion.
- Retrieval provider adapters.
- Merge planning.
- Token budgeting.
- Provenance preservation.
- Citation mapping.
- Metric calculation.
- Golden synthetic cases where Top-k fails and merging succeeds.

Core CI should run unit and golden tests. Full benchmark runs should be manual or scheduled separately.

## Claim Discipline

Do not claim MergeRAG universally wins. The project should say what happened under tested datasets, models, prompts, and token budgets.

Include visible limitations and failure cases. This makes the work more credible.
