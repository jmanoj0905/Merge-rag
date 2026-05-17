from mergerag.core.models import Chunk
from mergerag.merge.planner import plan


def _chunk(id_: str, rank: int, emb: list[float] | None = None) -> Chunk:
    return Chunk(
        id=id_, doc_id="d1", text="t", score=1.0 - rank * 0.05,
        rank=rank, embedding=emb or [1.0, 0.0],
    )


def _chunks(n: int) -> list[Chunk]:
    return [_chunk(f"c{i}", i) for i in range(n)]


def test_symmetric_plan_produces_ops():
    chunks = _chunks(10)
    result = plan(chunks, strategy="symmetric", strong_k=3)
    assert len(result.operations) > 0
    assert all(op.type == "symmetric" for op in result.operations)


def test_asymmetric_plan_produces_ops():
    chunks = _chunks(10)
    result = plan(chunks, strategy="asymmetric", strong_k=3)
    assert len(result.operations) > 0
    assert all(op.type == "asymmetric" for op in result.operations)


def test_strong_chunks_not_in_symmetric_ops():
    chunks = _chunks(6)
    strong_k = 2
    result = plan(chunks, strategy="symmetric", strong_k=strong_k)
    strong_ids = {c.id for c in chunks[:strong_k]}
    for op in result.operations:
        assert op.primary.id not in strong_ids
        assert op.secondary.id not in strong_ids


def test_asymmetric_primary_is_strong_chunk():
    chunks = _chunks(6)
    strong_k = 2
    result = plan(chunks, strategy="asymmetric", strong_k=strong_k)
    strong_ids = {c.id for c in chunks[:strong_k]}
    for op in result.operations:
        assert op.primary.id in strong_ids


def test_asymmetric_plan_respects_max_ops():
    chunks = _chunks(10)
    result = plan(chunks, strategy="asymmetric", strong_k=3, asymmetric_max_ops=1)
    assert len(result.operations) == 1


def test_asymmetric_plan_allows_zero_max_ops():
    chunks = _chunks(10)
    result = plan(chunks, strategy="asymmetric", strong_k=3, asymmetric_max_ops=0)
    assert result.operations == []


def test_fewer_chunks_than_strong_k_returns_empty_plan():
    chunks = _chunks(2)
    result = plan(chunks, strategy="symmetric", strong_k=5)
    assert result.operations == []
