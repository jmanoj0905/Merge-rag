# Benchmark Findings — HotpotQA Dev Distractor, n=500

**Run:** 2026-05-07  
**Dataset:** HotpotQA dev distractor set (10 context docs per question, only 2 supporting)  
**Model:** qwen2.5:3b via Ollama  
**Fixture:** `data/hotpotqa_dev_distractor.json`, first 500 examples  
**Results file:** `results/hotpot_dev_500.json`

## Results

| strategy   |   EM  |   F1  | latency_ms | tokens |   n |
|------------|-------|-------|------------|--------|-----|
| top_k      | 0.232 | 0.323 |       3836 |    380 | 500 |
| symmetric  | 0.224 | 0.309 |       5687 |    384 | 500 |
| asymmetric | 0.242 | 0.334 |       5447 |    391 | 500 |

Zero failures across all strategies.

## Key Findings

**Asymmetric beats top_k (+1.0 EM, +1.1 F1).** Augmenting strong anchor chunks with related weak context helps multi-hop questions. Cost: ~42% latency overhead, +11 tokens per call.

**Symmetric hurts vs top_k (-0.8 EM, -1.4 F1).** Pairing two weak same-doc chunks creates noisier merged context. The LLM does worse with the merged version despite ~48% latency overhead.

**Root cause of symmetric underperformance (hypothesis):** Two weak chunks from the same Wikipedia article are likely topically adjacent but not complementary. Concatenating them produces a longer, noisier passage rather than a tighter, more focused one. Needs investigation (see below).

## Open Questions

1. **Why does symmetric hurt?** Sample merged chunks from failed symmetric cases and inspect the merge output. The same-doc constraint is correct, but the merged text may be confusing the LLM -- e.g. two unrelated sentences from the same article concatenated together.

2. **Breakdown by question type.** HotpotQA has `bridge` (multi-hop) and `comparison` question types. Asymmetric likely dominates on bridge; comparison questions may behave differently. Slice `results/hotpot_dev_500.json` by `type` field.

3. **Reduce asymmetric latency.** The ~1600ms overhead over top_k is merge + LLM overhead. Profile the executor to see where time goes.

## Context on Fixes Applied Before This Run

Both merge strategies had a cross-doc pairing bug: chunks were paired by cosine similarity across the entire shared index with no document boundary constraint. On a multi-doc corpus (HotpotQA has 10 docs per question), this caused unrelated chunks from different documents to be merged.

Fix applied to both `asymmetric.py` and `symmetric.py`:
- `same_doc_only=True` -- only pair chunks sharing the same `doc_id`
- `min_similarity=0.3` -- skip pairs below cosine similarity threshold

Confirmed root cause from tiny-fixture run: asymmetric merged a Guido van Rossum chunk (from Python question context) with a Wikipedia anchor, causing the LLM to answer "Haarlem" instead of "Huntsville" for "In what city was the founder of Wikipedia born?"
