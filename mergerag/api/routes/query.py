import chromadb
from chromadb.errors import NotFoundError as ChromaNotFoundError
from fastapi import APIRouter, Depends, HTTPException, Request

from mergerag.api.config import Settings
from mergerag.api.deps import get_pipeline, get_settings_dep
from mergerag.api.schemas import (
    ChunkOut,
    CitationOut,
    MergedChunkOut,
    MergeOpOut,
    MergePlanOut,
    QueryRequest,
    QueryResponse,
)
from mergerag.core.models import Chunk, MergedChunk

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> QueryResponse:
    if settings.chroma_persist_path:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_path)
    else:
        _client = chromadb.EphemeralClient()
    try:
        _client.get_collection(body.collection_name)
    except (ValueError, ChromaNotFoundError):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{body.collection_name}' not found",
        )

    pipeline = get_pipeline(body.collection_name, body.params, request, settings)
    trace = pipeline.run(body.query, body.strategy)

    retrieved_chunks = [
        ChunkOut(
            type="chunk",
            id=c.id,
            doc_id=c.doc_id,
            text=c.text,
            score=c.score,
            rank=c.rank,
        )
        for c in trace.retrieved_chunks
    ]

    merged_chunks = [
        MergedChunkOut(
            type="merged",
            id=m.id,
            text=m.text,
            score=m.score,
            source_chunk_ids=m.source_chunk_ids,
            merge_type=m.merge_type,
        )
        for m in trace.merged_chunks
    ]

    final_context = []
    for item in trace.final_context:
        if isinstance(item, Chunk):
            final_context.append(
                ChunkOut(
                    type="chunk",
                    id=item.id,
                    doc_id=item.doc_id,
                    text=item.text,
                    score=item.score,
                    rank=item.rank,
                )
            )
        else:
            final_context.append(
                MergedChunkOut(
                    type="merged",
                    id=item.id,
                    text=item.text,
                    score=item.score,
                    source_chunk_ids=item.source_chunk_ids,
                    merge_type=item.merge_type,
                )
            )

    citations = [
        CitationOut(sentence=c.sentence, chunk_ids=c.chunk_ids)
        for c in trace.citations
    ]

    if trace.merge_plan is None:
        merge_plan = None
    else:
        merge_plan = MergePlanOut(
            operations=[
                MergeOpOut(
                    type=op.type,
                    primary_id=op.primary.id,
                    secondary_id=op.secondary.id,
                )
                for op in trace.merge_plan.operations
            ]
        )

    return QueryResponse(
        query=trace.query,
        strategy=trace.strategy,
        answer=trace.answer,
        citations=citations,
        token_count=trace.token_count,
        latency_ms=trace.latency_ms,
        retrieved_chunks=retrieved_chunks,
        merged_chunks=merged_chunks,
        final_context=final_context,
        merge_plan=merge_plan,
    )
