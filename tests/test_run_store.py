from datetime import datetime, timezone, timedelta
import pytest
from mergerag.adapters.run_store import SQLiteRunStore
from mergerag.core.models import (
    Chunk, MergedChunk, MergePlan, MergeOp, Citation, RunTrace,
)


def _make_trace(**overrides) -> RunTrace:
    chunk = Chunk(id="c1", doc_id="d1", text="hello world", score=0.9, rank=0, embedding=[0.1, 0.2])
    defaults: dict = dict(
        query="test query",
        strategy="top_k",
        retrieved_chunks=[chunk],
        merge_plan=None,
        merged_chunks=[],
        final_context=[chunk],
        answer="The answer.",
        citations=[Citation(sentence="The answer.", chunk_ids=["c1"])],
        token_count=5,
        latency_ms=42.0,
        collection_name="test_col",
        config={"top_n": 20, "top_k": 5, "strong_k": 5, "token_budget": 2048},
    )
    defaults.update(overrides)
    return RunTrace(**defaults)


@pytest.fixture
def store(tmp_path):
    return SQLiteRunStore(str(tmp_path / "runs.db"))


def test_save_and_get_scalar_fields(store):
    trace = _make_trace()
    store.save(trace)

    result = store.get(trace.run_id)
    assert result is not None
    assert result.run_id == trace.run_id
    assert result.query == "test query"
    assert result.strategy == "top_k"
    assert result.answer == "The answer."
    assert result.token_count == 5
    assert result.latency_ms == pytest.approx(42.0)
    assert result.collection_name == "test_col"
    assert result.config == {"top_n": 20, "top_k": 5, "strong_k": 5, "token_budget": 2048}


def test_get_nonexistent_returns_none(store):
    assert store.get("nonexistent-run-id") is None


def test_round_trip_preserves_retrieved_chunks(store):
    trace = _make_trace()
    store.save(trace)
    result = store.get(trace.run_id)

    assert len(result.retrieved_chunks) == 1
    chunk = result.retrieved_chunks[0]
    assert chunk.id == "c1"
    assert chunk.doc_id == "d1"
    assert chunk.text == "hello world"
    assert chunk.score == pytest.approx(0.9)
    assert chunk.rank == 0


def test_round_trip_strips_embeddings(store):
    trace = _make_trace()
    store.save(trace)
    result = store.get(trace.run_id)

    assert result.retrieved_chunks[0].embedding == []


def test_round_trip_preserves_citations(store):
    trace = _make_trace()
    store.save(trace)
    result = store.get(trace.run_id)

    assert len(result.citations) == 1
    assert result.citations[0].sentence == "The answer."
    assert result.citations[0].chunk_ids == ["c1"]


def test_round_trip_null_merge_plan(store):
    trace = _make_trace(merge_plan=None)
    store.save(trace)
    result = store.get(trace.run_id)
    assert result.merge_plan is None


def test_round_trip_with_merge_plan(store):
    c1 = Chunk(id="c1", doc_id="d1", text="a", score=0.9, rank=0)
    c2 = Chunk(id="c2", doc_id="d1", text="b", score=0.5, rank=1)
    m = MergedChunk(id="m1", text="merged", score=0.7, source_chunk_ids=["c1", "c2"], merge_type="symmetric")
    plan = MergePlan(operations=[MergeOp(type="symmetric", primary=c1, secondary=c2)])

    trace = _make_trace(
        strategy="symmetric",
        retrieved_chunks=[c1, c2],
        merge_plan=plan,
        merged_chunks=[m],
        final_context=[m],
    )
    store.save(trace)
    result = store.get(trace.run_id)

    assert result.strategy == "symmetric"
    assert result.merge_plan is not None
    assert len(result.merge_plan.operations) == 1
    assert result.merge_plan.operations[0].type == "symmetric"
    assert result.merge_plan.operations[0].primary.id == "c1"
    assert result.merge_plan.operations[0].secondary.id == "c2"
    assert len(result.merged_chunks) == 1
    assert result.merged_chunks[0].id == "m1"
    assert result.merged_chunks[0].source_chunk_ids == ["c1", "c2"]
    assert result.merged_chunks[0].merge_type == "symmetric"


def test_round_trip_final_context_mixed_types(store):
    chunk = Chunk(id="c1", doc_id="d1", text="raw", score=0.9, rank=0)
    merged = MergedChunk(id="m1", text="merged", score=0.7, source_chunk_ids=["c2", "c3"], merge_type="asymmetric")

    trace = _make_trace(final_context=[chunk, merged])
    store.save(trace)
    result = store.get(trace.run_id)

    assert len(result.final_context) == 2
    assert isinstance(result.final_context[0], Chunk)
    assert isinstance(result.final_context[1], MergedChunk)
    assert result.final_context[0].id == "c1"
    assert result.final_context[1].id == "m1"


def test_round_trip_preserves_created_at(store):
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    trace = _make_trace(created_at=fixed_time)
    store.save(trace)
    result = store.get(trace.run_id)

    assert result.created_at == fixed_time


def test_list_runs_returns_all_saved(store):
    for i in range(3):
        store.save(_make_trace(query=f"query {i}"))

    runs = store.list_runs()
    assert len(runs) == 3


def test_list_runs_ordered_newest_first(store):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        store.save(_make_trace(query=f"query {i}", created_at=base + timedelta(seconds=i)))

    runs = store.list_runs()
    assert runs[0].query == "query 2"
    assert runs[1].query == "query 1"
    assert runs[2].query == "query 0"


def test_list_runs_pagination(store):
    for i in range(5):
        store.save(_make_trace(query=f"query {i}"))

    page1 = store.list_runs(limit=2, offset=0)
    page2 = store.list_runs(limit=2, offset=2)
    page3 = store.list_runs(limit=2, offset=4)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1
    all_ids = {r.run_id for r in page1 + page2 + page3}
    assert len(all_ids) == 5


def test_list_runs_empty(store):
    assert store.list_runs() == []
