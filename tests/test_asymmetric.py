from mergerag.core.models import Chunk
from mergerag.merge.strategies.asymmetric import assign_to_anchors


def _chunk(id_: str, emb: list[float], rank: int = 5) -> Chunk:
    return Chunk(id=id_, doc_id="d1", text="t", score=0.3, rank=rank, embedding=emb)


def test_weak_assigned_to_nearest_anchor():
    strong = [
        _chunk("s1", [1.0, 0.0], rank=0),
        _chunk("s2", [0.0, 1.0], rank=1),
    ]
    weak = [_chunk("w1", [0.9, 0.1], rank=5)]  # closer to s1
    ops = assign_to_anchors(weak, strong)
    assert len(ops) == 1
    assert ops[0].primary.id == "s1"
    assert ops[0].secondary.id == "w1"


def test_all_ops_are_asymmetric_type():
    strong = [_chunk("s1", [1.0, 0.0], rank=0)]
    weak = [_chunk(f"w{i}", [float(i), 0.0], rank=i + 5) for i in range(3)]
    ops = assign_to_anchors(weak, strong)
    assert all(op.type == "asymmetric" for op in ops)


def test_empty_weak_returns_no_ops():
    strong = [_chunk("s1", [1.0, 0.0], rank=0)]
    assert assign_to_anchors([], strong) == []


def test_empty_strong_returns_no_ops():
    weak = [_chunk("w1", [1.0, 0.0], rank=5)]
    assert assign_to_anchors(weak, []) == []


def test_each_weak_chunk_gets_one_op():
    strong = [_chunk("s1", [1.0, 0.0], rank=0)]
    weak = [_chunk(f"w{i}", [1.0, 0.0], rank=i + 5) for i in range(4)]
    ops = assign_to_anchors(weak, strong)
    assert len(ops) == 4
