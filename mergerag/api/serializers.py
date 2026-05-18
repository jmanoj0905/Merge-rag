from __future__ import annotations

from mergerag.api.schemas import (
    ChunkOut,
    CitationOut,
    ContextItemOut,
    MergedChunkOut,
    MergeOpOut,
    MergePlanOut,
    RunDetail,
)
from mergerag.core.models import Chunk, MergedChunk, MergePlan, RunTrace


def chunk_out(chunk: Chunk) -> ChunkOut:
    return ChunkOut(
        type="chunk",
        id=chunk.id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        score=chunk.score,
        rank=chunk.rank,
    )


def merged_chunk_out(chunk: MergedChunk) -> MergedChunkOut:
    return MergedChunkOut(
        type="merged",
        id=chunk.id,
        text=chunk.text,
        score=chunk.score,
        source_chunk_ids=chunk.source_chunk_ids,
        merge_type=chunk.merge_type,
    )


def context_item_out(item: Chunk | MergedChunk) -> ContextItemOut:
    if isinstance(item, Chunk):
        return chunk_out(item)
    return merged_chunk_out(item)


def merge_plan_out(plan: MergePlan | None) -> MergePlanOut | None:
    if plan is None:
        return None
    return MergePlanOut(
        operations=[
            MergeOpOut(
                type=op.type,
                primary_id=op.primary.id,
                secondary_id=op.secondary.id,
            )
            for op in plan.operations
        ]
    )


def citation_outs(trace: RunTrace) -> list[CitationOut]:
    return [
        CitationOut(sentence=citation.sentence, chunk_ids=citation.chunk_ids)
        for citation in trace.citations
    ]


def run_detail_out(trace: RunTrace) -> RunDetail:
    return RunDetail(
        run_id=trace.run_id,
        created_at=trace.created_at.isoformat(),
        query=trace.query,
        strategy=trace.strategy,
        collection_name=trace.collection_name,
        config=trace.config,
        answer=trace.answer,
        citations=citation_outs(trace),
        token_count=trace.token_count,
        latency_ms=trace.latency_ms,
        retrieved_chunks=[chunk_out(chunk) for chunk in trace.retrieved_chunks],
        merged_chunks=[merged_chunk_out(chunk) for chunk in trace.merged_chunks],
        final_context=[context_item_out(item) for item in trace.final_context],
        merge_plan=merge_plan_out(trace.merge_plan),
    )
