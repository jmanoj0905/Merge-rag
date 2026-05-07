from mergerag.core.models import Chunk
from mergerag.adapters.retriever import ChromaRetriever


def _make_retriever() -> ChromaRetriever:
    return ChromaRetriever(collection_name="test_col")


def _chunks_and_embeddings() -> tuple[list[Chunk], list[list[float]]]:
    chunks = [
        Chunk(id="c1", doc_id="d1", text="Paris is the capital of France.", score=0.0, rank=0),
        Chunk(id="c2", doc_id="d1", text="Berlin is the capital of Germany.", score=0.0, rank=1),
        Chunk(id="c3", doc_id="d1", text="Rome is the capital of Italy.", score=0.0, rank=2),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return chunks, embeddings


def test_index_and_retrieve_returns_chunks():
    r = _make_retriever()
    chunks, embeddings = _chunks_and_embeddings()
    r.index(chunks, embeddings)
    results = r.retrieve(query_embedding=[1.0, 0.0, 0.0], top_n=2)
    assert len(results) == 2


def test_retrieve_sorted_by_score_descending():
    r = _make_retriever()
    chunks, embeddings = _chunks_and_embeddings()
    r.index(chunks, embeddings)
    results = r.retrieve(query_embedding=[1.0, 0.0, 0.0], top_n=3)
    scores = [c.score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieved_chunks_have_embeddings():
    r = _make_retriever()
    chunks, embeddings = _chunks_and_embeddings()
    r.index(chunks, embeddings)
    results = r.retrieve(query_embedding=[1.0, 0.0, 0.0], top_n=1)
    assert len(results[0].embedding) == 3


def test_retrieved_chunks_have_correct_rank():
    r = _make_retriever()
    chunks, embeddings = _chunks_and_embeddings()
    r.index(chunks, embeddings)
    results = r.retrieve(query_embedding=[1.0, 0.0, 0.0], top_n=3)
    assert results[0].rank == 0
    assert results[1].rank == 1
    assert results[2].rank == 2
