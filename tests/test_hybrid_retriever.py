import pytest
from mergerag.core.models import Chunk, Query
from mergerag.adapters.retriever import rrf_fuse, BM25Index, HybridRetriever


def _chunk(id_: str, rank: int = 0) -> Chunk:
    return Chunk(
        id=id_, doc_id="d1", text=f"text about {id_}",
        score=0.9 - rank * 0.1, rank=rank, embedding=[],
    )


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


# ── BM25Index ─────────────────────────────────────────────────────────────────

def test_bm25_index_retrieves_by_keyword():
    chunks = [
        Chunk(id="c_fox", doc_id="d1", text="the quick brown fox", score=0.0, rank=0, embedding=[]),
        Chunk(id="c_cat", doc_id="d1", text="the lazy cat sleeps", score=0.0, rank=1, embedding=[]),
    ]
    idx = BM25Index(chunks)
    results = idx.retrieve("fox", top_n=1)
    assert results[0].id == "c_fox"


def test_bm25_index_empty_corpus_returns_empty():
    idx = BM25Index([])
    results = idx.retrieve("anything", top_n=5)
    assert results == []


def test_bm25_index_top_n_limits_results():
    chunks = [
        Chunk(id=f"c{i}", doc_id="d1", text=f"word{i} common", score=0.0, rank=i, embedding=[])
        for i in range(10)
    ]
    idx = BM25Index(chunks)
    results = idx.retrieve("common", top_n=3)
    assert len(results) == 3


def test_bm25_index_rare_term_ranks_highest():
    chunks = [
        Chunk(id="rare", doc_id="d1", text="derrickson unique filmmaker", score=0.0, rank=0, embedding=[]),
        Chunk(id="common1", doc_id="d1", text="american filmmaker director", score=0.0, rank=1, embedding=[]),
        Chunk(id="common2", doc_id="d1", text="filmmaker director producer", score=0.0, rank=2, embedding=[]),
    ]
    idx = BM25Index(chunks)
    results = idx.retrieve("derrickson", top_n=3)
    assert results[0].id == "rare"


def test_bm25_index_normalizes_punctuation():
    chunks = [
        Chunk(id="match", doc_id="d1", text="Scott Derrickson's nationality is American.", score=0.0, rank=0, embedding=[]),
        Chunk(id="other", doc_id="d1", text="A different filmmaker biography.", score=0.0, rank=1, embedding=[]),
    ]
    idx = BM25Index(chunks)
    results = idx.retrieve("Derrickson nationality?", top_n=1)
    assert results[0].id == "match"


# ── HybridRetriever ───────────────────────────────────────────────────────────

def test_hybrid_retriever_rebuilds_bm25_after_index():
    r = HybridRetriever(collection_name="test_bm25_rebuild")
    chunks = [
        Chunk(id="c1", doc_id="d1", text="unique xylophone text", score=0.0, rank=0, embedding=[])
    ]
    r.index(chunks, embeddings=[[1.0, 0.0, 0.0]])
    bm25_texts = [c.text for c in r._bm25._chunks]
    assert "unique xylophone text" in bm25_texts


def test_hybrid_retriever_surfaces_keyword_match():
    """Chunk matching by keyword (BM25) but not by vector (dense) must appear in results."""
    r = HybridRetriever(collection_name="test_hybrid_kw")
    chunks = [
        Chunk(id="c_dense", doc_id="d1", text="the quick brown fox jumps over", score=0.0, rank=0, embedding=[]),
        Chunk(id="c_sparse", doc_id="d1", text="derrickson is an american filmmaker", score=0.0, rank=1, embedding=[]),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],  # matches query embedding
        [0.0, 0.0, 1.0],  # does NOT match query embedding
    ]
    r.index(chunks, embeddings)

    # Query embedding points toward c_dense; query text matches c_sparse via BM25
    query = Query(text="derrickson nationality", embedding=[1.0, 0.0, 0.0])
    results = r.retrieve(query, top_n=2)

    result_ids = [c.id for c in results]
    assert "c_sparse" in result_ids  # BM25 arm surfaces the keyword match


def test_hybrid_retriever_returns_top_n_results():
    r = HybridRetriever(collection_name="test_hybrid_topn")
    chunks = [
        Chunk(id=f"c{i}", doc_id="d1", text=f"document about topic {i}", score=0.0, rank=i, embedding=[])
        for i in range(6)
    ]
    embeddings = [[float(i == j) for j in range(6)] for i in range(6)]
    r.index(chunks, embeddings)

    query = Query(text="topic 0", embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = r.retrieve(query, top_n=3)
    assert len(results) == 3


def test_hybrid_retriever_results_have_sequential_ranks():
    r = HybridRetriever(collection_name="test_hybrid_ranks")
    chunks = [
        Chunk(id="a", doc_id="d1", text="alpha retrieval", score=0.0, rank=0, embedding=[]),
        Chunk(id="b", doc_id="d1", text="beta document", score=0.0, rank=1, embedding=[]),
    ]
    r.index(chunks, embeddings=[[1.0, 0.0], [0.0, 1.0]])

    query = Query(text="alpha", embedding=[1.0, 0.0])
    results = r.retrieve(query, top_n=2)
    assert [c.rank for c in results] == [0, 1]
