from mergerag.adapters.embedder import SentenceTransformerEmbedder


def test_embed_returns_one_vector_per_text():
    emb = SentenceTransformerEmbedder()
    results = emb.embed(["hello", "world", "test"])
    assert len(results) == 3


def test_embed_vectors_have_nonzero_length():
    emb = SentenceTransformerEmbedder()
    results = emb.embed(["hello"])
    assert len(results[0]) > 0


def test_embed_returns_floats():
    emb = SentenceTransformerEmbedder()
    results = emb.embed(["hello"])
    assert all(isinstance(v, float) for v in results[0])


def test_different_texts_produce_different_embeddings():
    emb = SentenceTransformerEmbedder()
    a, b = emb.embed(["cat", "quantum mechanics"])
    assert a != b
