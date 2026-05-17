# Benchmark Findings — HotpotQA Dev Distractor, n=500

**Dataset:** HotpotQA dev distractor set (10 context docs per question, only 2 supporting)  
**Model:** qwen2.5:3b (v1–v3), qwen2.5:7b (v4) via Ollama  
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

## Run v3 — 2026-05-16 (HybridRetriever: BM25 + dense, RRF k=60)

`results/hotpot_dev_500_hybrid.json`

| strategy   |   EM  |   F1  | latency_ms | tokens |   n |
|------------|-------|-------|------------|--------|-----|
| top_k      | 0.240 | 0.320 |       3833 |    395 | 500 |
| symmetric  | 0.240 | 0.320 |       3219 |    406 | 500 |
| asymmetric | 0.240 | 0.320 |       3495 |    404 | 500 |

All three strategies flat at 0.240 EM / 0.320 F1. Hybrid retrieval did not improve over dense-only (v2: 0.244–0.252 EM).

**Why hybrid didn't help at aggregate level.** RRF over-fetches from both arms (top_n × 2 candidates each) then clips to top_n. The BM25 arm introduces keyword-matched candidates that displace high-precision dense results more often than they repair entity recall gaps. HotpotQA's distractor paragraphs are topically similar (10 docs per question), so BM25 tends to surface plausible-but-wrong paragraphs that share surface terms with the query.

**BM25 arm does fix targeted entity recall.** Smoke test confirmed: "Scott Derrickson nationality" query surfaces `Scott_Derrickson-0000` ("is an American director") at BM25 rank 2, which the dense arm misses. The gain is real but too infrequent (~few questions in 500) to move aggregate EM.

**Next step candidates.** Tune `bm25_candidates` (currently `top_n × 2`) — a tighter BM25 pool (e.g. top_n only) may reduce noise. Alternatively, only activate the BM25 arm for bridge questions where entity recall matters most.

---

## Run v4 — 2026-05-17 (qwen2.5:7b upgrade, n=100)

`results/hotpot_dev_100_cot_7b.json`

| strategy   |   EM  |   F1  | latency_ms | tokens |   n |
|------------|-------|-------|------------|--------|-----|
| top_k      | 0.20  | 0.27  |       4228 |    398 | 100 |
| symmetric  | 0.23  | 0.30  |       3691 |    417 | 100 |
| asymmetric | 0.22  | 0.26  |       4477 |    410 | 100 |

n=100 noise ±0.05. Aggregate roughly flat vs 3b baseline. Model swap alone not a clear win at this n.

**Motivating case.** For `"Were Scott Derrickson and Ed Wood of the same nationality?"` (gold: `yes`), the 3b model answered `No` even with both relevant chunks in context. Direct inspection showed 3b contradicting itself: "No, ... Scott Derrickson is American, while Ed Wood was also American." Pure reasoning failure — retrieval and prompt were correct.

**7b on clean 2-chunk context** → `Yes` ✓.  
**7b on full pipeline context (8 chunks with `John_Scott_(ice_hockey)` Canadian distractor)** → `No` ✗.

Distractors with overlapping surnames (multiple "Scotts" in the HotpotQA pool) confuse even 7b. Scott_Derrickson surfaces at hybrid rank 6, so the default `top_k=5` cuts it off and forces a wrong answer regardless of model size.

### CoT prompt experiment (reverted)

Attempted an entity-grounded chain-of-thought template (reason per entity, then `Final: <answer>`). Fixed Scott_Derrickson on a manual run, but n=100 sweep showed regressions:
- Model copied entity names as final answers (`Janet_Waldo` instead of `Chief of Protocol`).
- Citation parsing broke when 7b omitted brackets in the CoT format.
- Aggregate EM dropped vs the non-CoT baseline.

Reverted prompt and answer-extraction logic; kept the 7b model upgrade. Scott_Derrickson-style cases (entity disambiguation under surname-overlap distractors, with the correct chunk beyond default `top_k`) remain a known limitation. Workarounds: bump `top_k` to 7+ via the UI sliders, or use the hybrid retriever for entity-name queries.

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
