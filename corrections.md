# Code-review corrections

Findings from external review. Status reflects state after commit `5c74460c` ("fix Chroma lifecycle, secure ingestion, and optimize pipeline").

Overall this is a reasonably structured small RAG project: the domain models, ports, adapters, merge planner/executor, FastAPI routes, Streamlit client, scripts, and tests are separated in a way that is easy to follow. The test coverage is also better than many prototype projects. The main corrections are around runtime correctness and production hardening rather than basic organization.

---

✅ **Chroma lifecycle (FIXED — 5c74460c).** The previous default of `chroma_persist_path=None` created a fresh `chromadb.EphemeralClient()` inside each `ChromaRetriever`, and the query route created a separate ephemeral client just to check collection existence — so `/ingest` would index into one in-memory client while `/query` checked a different one. Now a single client is created in FastAPI `lifespan` (`make_chroma_client` in `mergerag/api/chroma.py`), stored on `app.state.chroma_client`, and injected into every `ChromaRetriever` / `HybridRetriever` via the `client=` constructor arg. Query, ingest, and collections routes all reuse it through `get_chroma_client` in `deps.py`.

✅ **Hybrid retriever cache staleness (FIXED — 5c74460c).** `get_pipeline()` still caches `HybridRetriever` instances by `(collection, persist_path)`, but `/ingest` now calls `refresh_hybrid_cache(collection_name, persist_path)` after a successful ingest, which calls `cached.refresh_sparse_index()` to rebuild the BM25 index from the updated Chroma collection. `DELETE /collections/{name}` calls `clear_hybrid_cache(...)` to drop the cached instance. `HybridRetriever.retrieve()` also detects collection-count drift between calls (`current_count != self._bm25_source_count`) and rebuilds defensively.

✅ **API request parameter validation (FIXED — 5c74460c).** `PipelineParams` in `schemas.py` now uses `PositiveInt` for `top_n`/`top_k`/`strong_k`/`token_budget`, `Field(ge=0)` for `asymmetric_max_ops`, and a `model_validator(mode="after")` for the cross-field constraints `top_k <= top_n` and `strong_k <= top_n`. `get_pipeline` in `deps.py` re-runs the resolved-value check (`_validate_resolved_params`) after merging request params with `Settings` defaults, returning 422 on violation.

✅ **`token_budget` enforcement (FIXED — 5c74460c).** `MergeRAGPipeline._within_token_budget` is now called on both the `top_k` branch and the post-rerank pool of the merge branches in `pipeline.py`. It packs items in order until the next item would exceed the budget (using the same `len(text.split())` approximation as `_count_tokens`). `token_budget <= 0` is treated as "no limit" so existing tests pass when callers want the full pool.

✅ **Duplicated Chroma client logic in `query` route (FIXED — 5c74460c).** The collection-existence probe in `routes/query.py` now uses the shared `app.state.chroma_client` via `get_chroma_client(request)` — there is no more standalone ephemeral client allocated inside the route handler.

✅ **Response mapping duplication (FIXED — 5c74460c).** `mergerag/api/serializers.py` now holds `chunk_out`, `merged_chunk_out`, `context_item_out`, `merge_plan_out`, `citation_outs`, and `run_detail_out`. `routes/query.py` and `routes/runs.py` both import from there instead of converting schemas inline.

✅ **Re-ingest of same `doc_id` (FIXED — 5c74460c).** `ChromaRetriever.index()` now calls `self._collection.upsert(...)`, so re-running ingestion against deterministic chunk IDs (`{doc_id}-0000`) overwrites instead of duplicating or raising.

✅ **Unbounded upload size on `/ingest` (FIXED — 5c74460c).** `routes/ingest.py` now reads at most `MAX_UPLOAD_BYTES + 1` (default 10 MB, configurable via `Settings.max_upload_bytes`) and returns 413 on overflow. The route also whitelists `.txt` and `.md` and returns 422 on unsupported extensions or missing `doc_id`.

✅ **Rerank pool dropped unmerged weak chunks (FIXED — 5c74460c).** The merge branch in `pipeline.py` now reranks `chunks[:strong_k] + merged + unmerged` (where `unmerged = chunks[strong_k:]` minus IDs already consumed by merge ops). Cosine similarity decides what makes the final `top_k`.

✅ **Citation parser trusted any bracketed text (FIXED — 5c74460c).** `_parse_citations(answer, allowed_ids)` now filters every bracketed token against the set of IDs present in `final_context`. Brackets with no overlap are dropped; sentences with no valid citations are omitted from the returned list.

✅ **BM25 tokenization (FIXED — 5c74460c).** `mergerag/adapters/retriever.py` defines a module-level `_tokenize(text) = re.findall(r"[a-z0-9]+", text.lower())` shared by `BM25Index` indexing and querying. Punctuation no longer sticks to entity tokens, which materially helps the HotpotQA entity-name use case.

✅ **Runtime artifacts in source control (FIXED — 5c74460c).** `.gitignore` now excludes `runs.db`, `runs.db-*`, and `data/chroma/`. Benchmark result JSON files under `results/` are kept intentionally so the published numbers remain reproducible from a checkout.

⏳ **Error handling around external services (OPEN).** Ollama failures and SentenceTransformer model-load failures still surface as generic 5xx. `routes/query.py` now wraps the pipeline run in `try/except` and returns 502 on adapter failure, but there is no per-adapter classification (timeout vs. unreachable vs. malformed response) and no retry/backoff.

⏳ **Synchronous request path (OPEN).** Embedding, Chroma calls, and Ollama generation still run synchronously inside FastAPI's threadpool. Acceptable for the local demo / Streamlit dashboard, but concurrent use would benefit from request timeouts, cancellation, and a bounded worker queue. The benchmark already runs out-of-band via `scripts/benchmark_cli.py`, so this is not blocking benchmark credibility.

⏳ **Minor maintainability smells (OPEN).** Some historical-benchmark explanations still live as comments inside production code, and `HybridRetriever` reaches into `self._chroma._collection.count()` via the public `collection_count()` accessor on `ChromaRetriever` (cleaner than direct private access, but still couples the two). Not urgent.
