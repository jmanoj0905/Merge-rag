import pytest
from mergerag.core.models import Chunk, Query
from mergerag.adapters.retriever import rrf_fuse, BM25Index, HybridRetriever


def _chunk(id_: str, rank: int = 0) -> Chunk:
    return Chunk(id=id_, doc_id="d1", text=f"text about {id_}", score=0.9 - rank * 0.1, rank=rank, embedding=[])


# ── rrf_fuse ──────────────────────────────────────────────────────────────────

def test_rrf_fuse_chunk_in_both_lists_ranks_first():
    shared = _chunk("shared", rank=0)
    dense_only = _chunk("dense_only", rank=1)
    sparse_only = _chunk("sparse_only", rank=0)

    result = rrf_fuse(dense=[shared, dense_only], sparse=[sparse_only, shared])
    assert result[0].id == "shared"


def test_rrf_fuse_empty_sparse_returns_dense_order():
    c1 = _chunk("c1", rank=0)
    c2 = _chunk("c2", rank=1)

    result = rrf_fuse(dense=[c1, c2], sparse=[])
    assert [r.id for r in result] == ["c1", "c2"]


def test_rrf_fuse_sets_rank_sequentially():
    c1 = _chunk("c1", rank=0)
    c2 = _chunk("c2", rank=1)

    result = rrf_fuse(dense=[c1, c2], sparse=[])
    assert result[0].rank == 0
    assert result[1].rank == 1


def test_rrf_fuse_sets_score_to_rrf_value():
    c1 = _chunk("c1", rank=0)

    result = rrf_fuse(dense=[c1], sparse=[c1], k=60)
    # c1 appears at rank 0 in both lists: score = 1/(60+0) + 1/(60+0) = 2/60
    assert abs(result[0].score - 2 / 60) < 1e-9


def test_rrf_fuse_does_not_mutate_inputs():
    c1 = _chunk("c1", rank=0)
    original_score = c1.score
    rrf_fuse(dense=[c1], sparse=[c1])
    assert c1.score == original_score
