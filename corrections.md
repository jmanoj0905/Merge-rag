Overall this is a reasonably structured small RAG project: the domain models, ports, adapters, merge planner/executor, FastAPI routes, Streamlit client, scripts, and tests are separated in a way that is easy to follow. The test coverage is also better than many prototype projects. The main corrections are around runtime correctness and production hardening rather than basic organization.

The biggest functional issue is Chroma lifecycle handling. The default `chroma_persist_path=None` creates a new `chromadb.EphemeralClient()` inside each `ChromaRetriever`, and the query route also creates a separate ephemeral client just to check collection existence. That means an `/ingest` request can successfully index into one in-memory client, then `/query` checks a different in-memory client and returns collection not found. Either make persistent Chroma required for the API, or create one app-scoped Chroma client in `lifespan` and inject/share it across ingest/query/retriever construction.

The hybrid retriever cache can go stale. `get_pipeline()` caches `HybridRetriever` instances by collection and persist path, but `/ingest` always indexes through a fresh `ChromaRetriever`, not the cached hybrid instance. If the user ingests after a hybrid retriever has already been built, the dense Chroma data may update but the cached BM25 sparse index will not rebuild. Invalidate `_HYBRID_CACHE` on ingest/delete, or route ingestion through the cached retriever when hybrid is enabled, or make `HybridRetriever.retrieve()` detect collection changes and rebuild BM25.

API request parameters need validation. `PipelineParams` accepts arbitrary integers, including `top_n=0`, negative `top_k`, `strong_k > top_n`, negative `token_budget`, and negative `asymmetric_max_ops`. Those values can cause Chroma errors, empty contexts, surprising merge behavior, or invalid config. Use Pydantic constraints like positive integers and add cross-field validation for `top_k <= top_n` and `strong_k <= top_n`.

`token_budget` is stored but not enforced. `MergeRAGPipeline` accepts it and returns it in `config`, but final context selection never trims by budget. This makes the API misleading and can overflow the answer prompt. Add budget-aware context packing after reranking/top-k selection, preferably using the same token approximation as `_count_tokens()` or a real tokenizer.

The query route has duplicated Chroma client logic and does a collection existence check outside the retriever abstraction. That check can disagree with the retriever client, especially with ephemeral clients. Prefer a shared `make_chroma_client()` helper or push collection-not-found handling into the retriever dependency creation.

The response mapping logic is duplicated in `query.py` and `routes/runs.py`. Both manually convert `Chunk`, `MergedChunk`, citations, and merge plans into schemas. Move this to a serializer helper so future response changes do not drift between live query responses and saved run responses.

Re-ingesting the same `doc_id` can collide with deterministic chunk IDs like `doc-0000`. `ChromaRetriever.index()` uses `collection.add()`, which is not an upsert-style API. Re-running ingestion for the same document is likely to fail or duplicate behavior depending on Chroma version. Use `upsert()`, delete previous IDs for a doc before adding, or generate versioned IDs.

The ingestion endpoint reads the full uploaded file into memory before writing to a temp file. That is fine for small demos, but it needs size limits and streaming/chunked writes for a real API. Add a maximum upload size and return a clean 413/422 instead of letting memory use grow unexpectedly.

The merge rerank pool for non-`top_k` strategies is only `chunks[:strong_k] + merged`. Any unmerged weak chunks are dropped entirely even if they were originally high enough to belong in the final top-k. That may be intentional for the experiment, but if the goal is best answer quality, rerank `strong + merged + unmerged_weak` and let scores decide.

The citation parser trusts any bracketed text in the LLM answer as chunk IDs. It should filter citations against IDs actually present in `final_context`; otherwise the API can return citations for non-existent chunks or arbitrary bracketed text.

BM25 tokenization is very naive: `.lower().split()` keeps punctuation attached and gives weak matching for entity-heavy questions. For the project’s stated HotpotQA/entity recall use case, use a small normalization tokenizer shared by indexing and querying.

The repository includes runtime artifacts in source control or at least in the working tree: `runs.db`, `data/chroma/*`, and benchmark result JSON files. Keep seed fixtures if needed, but generated SQLite/Chroma stores should normally be ignored or moved under an explicitly documented sample-data path. Current `git status` shows modified `runs.db` and `data/chroma/chroma.sqlite3`, which makes accidental commits likely.

Error handling around external services is thin. Ollama failures, SentenceTransformer model-load failures, Chroma query failures, and malformed uploaded files will bubble out as generic 500 errors. Wrap expected adapter failures at route boundaries and return actionable HTTP errors.

The code is mostly synchronous in request paths, including embedding, Chroma calls, and Ollama generation. FastAPI will run sync handlers in a threadpool, but long LLM calls can still tie up server resources. For local demos this is acceptable; for concurrent use, add request timeouts, cancellation behavior, and a bounded worker queue.

Imports and formatting are mostly clean, but there are some maintainability smells: comments that explain historical benchmark choices inside production code, private access to `self._chroma._collection` in `HybridRetriever`, and repeated schema conversion code. None are urgent, but cleaning them up would make future changes safer.
