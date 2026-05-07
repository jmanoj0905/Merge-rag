from mergerag.core.models import Chunk
from mergerag.merge.strategies.symmetric import pair_weak_chunks


def _chunk(id_: str, emb: list[float]) -> Chunk:
    return Chunk(id=id_, doc_id="d1", text="t", score=0.3, rank=5, embedding=emb)


def test_pairs_two_similar_chunks():
    a = _chunk("c1", [1.0, 0.0])
    b = _chunk("c2", [1.0, 0.0])  # identical direction -- most similar
    c = _chunk("c3", [0.0, 1.0])  # orthogonal to a
    ops = pair_weak_chunks([a, b, c])
    # a and b should be paired (highest similarity)
    assert len(ops) == 1
    ids = {ops[0].primary.id, ops[0].secondary.id}
    assert ids == {"c1", "c2"}


def test_empty_list_returns_no_ops():
    assert pair_weak_chunks([]) == []


def test_single_chunk_returns_no_ops():
    a = _chunk("c1", [1.0, 0.0])
    assert pair_weak_chunks([a]) == []


def test_all_ops_are_symmetric_type():
    chunks = [_chunk(f"c{i}", [float(i), 0.0]) for i in range(1, 5)]
    ops = pair_weak_chunks(chunks)
    assert all(op.type == "symmetric" for op in ops)


def test_no_chunk_used_twice():
    chunks = [_chunk(f"c{i}", [float(i % 2), float((i+1) % 2)]) for i in range(6)]
    ops = pair_weak_chunks(chunks)
    used = [id_ for op in ops for id_ in [op.primary.id, op.secondary.id]]
    assert len(used) == len(set(used))
