from typing import Any

from fastapi import HTTPException, Request
from mergerag.adapters.retriever import ChromaRetriever, HybridRetriever
from mergerag.core.ports import EmbedderPort, LLMPort, RetrieverPort
from mergerag.pipeline import MergeRAGPipeline
from mergerag.api.config import Settings, get_settings
from mergerag.api.schemas import PipelineParams

_HYBRID_CACHE: dict[tuple[str, str | None], HybridRetriever] = {}


def get_embedder(request: Request) -> EmbedderPort:
    if not hasattr(request.app.state, "embedder"):
        raise HTTPException(status_code=503, detail="Embedder not initialised")
    return request.app.state.embedder


def get_llm(request: Request) -> LLMPort:
    if not hasattr(request.app.state, "llm"):
        raise HTTPException(status_code=503, detail="LLM not initialised")
    return request.app.state.llm


def get_settings_dep() -> Settings:
    return get_settings()


def get_chroma_client(request: Request) -> Any:
    if not hasattr(request.app.state, "chroma_client"):
        raise HTTPException(status_code=503, detail="Chroma client not initialised")
    return request.app.state.chroma_client


def refresh_hybrid_cache(collection_name: str, persist_path: str | None) -> None:
    cached = _HYBRID_CACHE.get((collection_name, persist_path))
    if cached is not None:
        cached.refresh_sparse_index()


def clear_hybrid_cache(collection_name: str, persist_path: str | None) -> None:
    _HYBRID_CACHE.pop((collection_name, persist_path), None)


def _validate_resolved_params(top_n: int, top_k: int, strong_k: int) -> None:
    if top_k > top_n:
        raise HTTPException(status_code=422, detail="top_k must be less than or equal to top_n")
    if strong_k > top_n:
        raise HTTPException(status_code=422, detail="strong_k must be less than or equal to top_n")


def get_pipeline(
    collection_name: str,
    params: PipelineParams,
    request: Request,
    settings: Settings,
) -> MergeRAGPipeline:
    """Helper called by route handlers. Resolves params and builds the pipeline."""
    embedder = get_embedder(request)
    llm = get_llm(request)
    chroma_client = get_chroma_client(request)
    retriever_name = params.retriever or settings.default_retriever
    retriever: RetrieverPort
    if retriever_name == "hybrid":
        key = (collection_name, settings.chroma_persist_path)
        cached = _HYBRID_CACHE.get(key)
        if cached is None:
            cached = HybridRetriever(
                collection_name=collection_name,
                persist_path=settings.chroma_persist_path,
                client=chroma_client,
            )
            _HYBRID_CACHE[key] = cached
        retriever = cached
    else:
        retriever = ChromaRetriever(
            collection_name=collection_name,
            persist_path=settings.chroma_persist_path,
            client=chroma_client,
        )

    top_n = params.top_n if params.top_n is not None else settings.default_top_n
    top_k = params.top_k if params.top_k is not None else settings.default_top_k
    strong_k = params.strong_k if params.strong_k is not None else settings.default_strong_k
    token_budget = params.token_budget if params.token_budget is not None else settings.default_token_budget
    asymmetric_max_ops = (
        params.asymmetric_max_ops
        if params.asymmetric_max_ops is not None
        else settings.default_asymmetric_max_ops
    )
    _validate_resolved_params(top_n, top_k, strong_k)

    return MergeRAGPipeline(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        top_n=top_n,
        top_k=top_k,
        strong_k=strong_k,
        token_budget=token_budget,
        asymmetric_max_ops=asymmetric_max_ops,
    )
