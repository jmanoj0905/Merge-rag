from fastapi import APIRouter, HTTPException, Request

from mergerag.api.schemas import (
    ChunkOut, CitationOut, ContextItemOut, MergedChunkOut,
    MergeOpOut, MergePlanOut, RunDetail, RunSummary,
)
from mergerag.core.models import Chunk, RunTrace

router = APIRouter()


def _trace_to_detail(trace: RunTrace) -> RunDetail:
    retrieved = [
        ChunkOut(type="chunk", id=c.id, doc_id=c.doc_id, text=c.text, score=c.score, rank=c.rank)
        for c in trace.retrieved_chunks
    ]
    merged = [
        MergedChunkOut(
            type="merged", id=m.id, text=m.text, score=m.score,
            source_chunk_ids=m.source_chunk_ids, merge_type=m.merge_type,
        )
        for m in trace.merged_chunks
    ]
    final_context: list[ContextItemOut] = []
    for item in trace.final_context:
        if isinstance(item, Chunk):
            final_context.append(
                ChunkOut(type="chunk", id=item.id, doc_id=item.doc_id, text=item.text, score=item.score, rank=item.rank)
            )
        else:
            final_context.append(
                MergedChunkOut(
                    type="merged", id=item.id, text=item.text, score=item.score,
                    source_chunk_ids=item.source_chunk_ids, merge_type=item.merge_type,
                )
            )
    citations = [CitationOut(sentence=c.sentence, chunk_ids=c.chunk_ids) for c in trace.citations]
    merge_plan = None
    if trace.merge_plan is not None:
        merge_plan = MergePlanOut(
            operations=[
                MergeOpOut(type=op.type, primary_id=op.primary.id, secondary_id=op.secondary.id)
                for op in trace.merge_plan.operations
            ]
        )
    return RunDetail(
        run_id=trace.run_id,
        created_at=trace.created_at.isoformat(),
        query=trace.query,
        strategy=trace.strategy,
        collection_name=trace.collection_name,
        config=trace.config,
        answer=trace.answer,
        citations=citations,
        token_count=trace.token_count,
        latency_ms=trace.latency_ms,
        retrieved_chunks=retrieved,
        merged_chunks=merged,
        final_context=final_context,
        merge_plan=merge_plan,
    )


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> list[RunSummary]:
    run_store = request.app.state.run_store
    traces = run_store.list_runs(limit=limit, offset=offset)
    return [
        RunSummary(
            run_id=t.run_id,
            created_at=t.created_at.isoformat(),
            query=t.query,
            strategy=t.strategy,
            collection_name=t.collection_name,
            token_count=t.token_count,
            latency_ms=t.latency_ms,
        )
        for t in traces
    ]


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, request: Request) -> RunDetail:
    run_store = request.app.state.run_store
    trace = run_store.get(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _trace_to_detail(trace)
