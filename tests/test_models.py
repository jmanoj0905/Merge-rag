from mergerag.core.models import (
    Chunk, MergedChunk, MergeOp, MergePlan, Citation, RunTrace
)


def test_chunk_defaults():
    c = Chunk(id="c1", doc_id="d1", text="hello", score=0.9, rank=0)
    assert c.embedding == []


def test_merged_chunk_provenance():
    m = MergedChunk(
        id="m1",
        text="merged",
        score=0.8,
        source_chunk_ids=["c1", "c2"],
        merge_type="symmetric",
    )
    assert m.source_chunk_ids == ["c1", "c2"]
    assert m.merge_type == "symmetric"


def test_merge_plan_holds_ops():
    c1 = Chunk(id="c1", doc_id="d1", text="a", score=0.5, rank=2)
    c2 = Chunk(id="c2", doc_id="d1", text="b", score=0.4, rank=3)
    op = MergeOp(type="symmetric", primary=c1, secondary=c2)
    plan = MergePlan(operations=[op])
    assert len(plan.operations) == 1
    assert plan.operations[0].primary.id == "c1"


def test_citation_fields():
    cit = Citation(sentence="Paris is the capital.", chunk_ids=["c1"])
    assert cit.chunk_ids == ["c1"]


def test_run_trace_fields():
    trace = RunTrace(
        query="q",
        strategy="top_k",
        retrieved_chunks=[],
        merge_plan=None,
        merged_chunks=[],
        final_context=[],
        answer="ans",
        citations=[],
        token_count=10,
        latency_ms=123.4,
    )
    assert trace.strategy == "top_k"
    assert trace.merge_plan is None
