import logging

from chromadb.errors import NotFoundError as ChromaNotFoundError
from fastapi import APIRouter, Depends, HTTPException, Request

from mergerag.api.config import Settings
from mergerag.api.deps import get_chroma_client, get_pipeline, get_settings_dep
from mergerag.api.serializers import (
    chunk_out,
    citation_outs,
    context_item_out,
    merge_plan_out,
    merged_chunk_out,
)
from mergerag.api.schemas import (
    QueryRequest,
    QueryResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> QueryResponse:
    client = get_chroma_client(request)
    try:
        client.get_collection(body.collection_name)
    except (ValueError, ChromaNotFoundError):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{body.collection_name}' not found",
        )

    pipeline = get_pipeline(body.collection_name, body.params, request, settings)
    try:
        trace = pipeline.run(body.query, body.strategy, body.collection_name)
        request.app.state.run_store.save(trace)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed for collection %s", body.collection_name)
        raise HTTPException(status_code=502, detail="Query execution failed") from exc

    return QueryResponse(
        query=trace.query,
        strategy=trace.strategy,
        answer=trace.answer,
        citations=citation_outs(trace),
        token_count=trace.token_count,
        latency_ms=trace.latency_ms,
        retrieved_chunks=[chunk_out(chunk) for chunk in trace.retrieved_chunks],
        merged_chunks=[merged_chunk_out(chunk) for chunk in trace.merged_chunks],
        final_context=[context_item_out(item) for item in trace.final_context],
        merge_plan=merge_plan_out(trace.merge_plan),
    )
