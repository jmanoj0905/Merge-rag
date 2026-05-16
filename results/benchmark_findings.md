# Benchmark Findings — HotpotQA Dev Distractor, n=500

**Dataset:** HotpotQA dev distractor set (10 context docs per question, only 2 supporting)  
**Model:** qwen2.5:3b via Ollama  
**Fixture:** `data/hotpotqa_dev_distractor.json`, first 500 examples (404 bridge, 96 comparison)

---

## Run v1 — 2026-05-07 (baseline)

`results/hotpot_dev_500.json`

| strategy   |   EM  |   F1  | latency_ms | tokens |   n |
|------------|-------|-------|------------|--------|-----|
| top_k      | 0.232 | 0.323 |       3836 |    380 | 500 |
| symmetric  | 0.224 | 0.309 |       5687 |    384 | 500 |
| asymmetric | 0.242 | 0.334 |       5447 |    391 | 500 |

Symmetric used LLM synthesis (since fixed). Asymmetric had no `max_ops` cap (since fixed).

---

## Run v2 — 2026-05-13 (with fixes)

`results/hotpot_dev_500_v2.json` + `results/hotpot_dev_500_sym_fixed.json`

| strategy        |   EM  |   F1  | latency_ms | tokens |   n |
|-----------------|-------|-------|------------|--------|-----|
| top_k           | 0.244 | 0.336 |       3646 |    380 | 500 |
| symmetric†      | 0.246 | 0.336 |       3796 |    381 | 500 |
| asymmetric‡     | 0.252 | 0.346 |       4817 |    393 | 500 |

† concatenation fix (no LLM synthesis). ‡ max_ops=3 cap.

### Per Question Type (v2)

| type       | strategy   |   EM  |   F1  |   n |
|------------|------------|-------|-------|-----|
| bridge     | top_k      | 0.208 | 0.297 | 404 |
| bridge     | symmetric  | 0.191 | 0.289 | 404 |
| bridge     | asymmetric | 0.218 | 0.309 | 404 |
| comparison | top_k      | 0.396 | 0.500 |  96 |
| comparison | symmetric  | 0.385 | 0.477 |  96 |
| comparison | asymmetric | 0.396 | 0.505 |  96 |

---

## Profile Results — 2026-05-16 (n=100, max_ops=3)

`logs/profile_latency.log`

| strategy   | embed | retrieve | merge | answer | total |
|------------|-------|----------|-------|--------|-------|
| top_k      |  87ms |     11ms |   0ms | 3417ms | 3514ms |
| symmetric  | 100ms |      7ms |   0ms | 1245ms | 1364ms |
| asymmetric |  55ms |      8ms | 2744ms| 1686ms | 4535ms |

Asymmetric merge fires on 16% of questions. When active: merge mean=17150ms (3 LLM calls × ~5700ms). Overall average stays at 4535ms because 84% of questions skip merge.

---

## Key Findings

**Symmetric fix (+2.2 EM, -33% latency).** Replacing LLM synthesis with concatenation fixed the underperformance. Symmetric now ties top_k on EM (0.246 vs 0.244) at comparable latency.

**Asymmetric wins overall (+0.8 EM vs top_k v2).** Consistent across both runs. Cost: ~32% latency overhead.

**Bridge vs comparison split.** Asymmetric wins clearly on bridge (+1.0 EM over top_k). On comparison, top_k and asymmetric are tied (0.396 EM). Symmetric underperforms on both types.

**Root cause of original symmetric failure.** LLM synthesis of two weak same-doc chunks produced noisier context than raw chunks. Concatenation avoids the synthesis step entirely and matches or beats top_k.

**Asymmetric latency bounded by max_ops=3.** Without cap, merge could run unbounded LLM calls. With cap, worst case is 3 calls (~17s when active) but fires on only 16% of questions.

---

## Frontend Routing

Smart route toggle in Streamlit app routes by detected question type:
- bridge → asymmetric
- comparison → top_k

Keyword classifier accuracy: **87%** against ground-truth HotpotQA type labels (n=500). FP rate (bridge→top_k) reduced from 126→18 after keyword refinement. FN penalty is zero since top_k ≈ asymmetric on comparison.

---

## Context on Fixes Applied

Both merge strategies had a cross-doc pairing bug: chunks were paired by cosine similarity across the entire shared index with no document boundary constraint. On a multi-doc corpus (HotpotQA has 10 docs per question), this caused unrelated chunks from different documents to be merged.

Fix applied to both `asymmetric.py` and `symmetric.py`:
- `same_doc_only=True` — only pair chunks sharing the same `doc_id`
- `min_similarity=0.3` — skip pairs below cosine similarity threshold

Confirmed root cause from tiny-fixture run: asymmetric merged a Guido van Rossum chunk with a Wikipedia anchor, causing the LLM to answer "Haarlem" instead of "Huntsville" for "In what city was the founder of Wikipedia born?"
