import logging

import chromadb
from rank_bm25 import BM25Okapi
from mergerag.core.models import Chunk, Query
from mergerag.core.ports import RetrieverPort

logger = logging.getLogger(__name__)


class ChromaRetriever(RetrieverPort):
    def __init__(self, collection_name: str = "mergerag", persist_path: str | None = None):
        if persist_path:
            self._client = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"doc_id": c.doc_id} for c in chunks],
        )

    def retrieve(self, query: Query, top_n: int) -> list[Chunk]:
        results = self._collection.query(
            query_embeddings=[query.embedding],
            n_results=top_n,
            include=["documents", "embeddings", "metadatas", "distances"],
        )
        chunks: list[Chunk] = []
        for i, (id_, doc, emb, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["embeddings"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # cosine distance = 1 - cosine_similarity, so similarity = 1 - distance
            score = 1.0 - float(dist)
            chunks.append(Chunk(
                id=id_,
                doc_id=meta["doc_id"],
                text=doc,
                score=score,
                rank=i,
                embedding=list(emb),
            ))
        return chunks


def rrf_fuse(
    dense: list[Chunk],
    sparse: list[Chunk],
    k: int = 60,
) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, chunk in enumerate(dense):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(sparse):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores, key=lambda id_: scores[id_], reverse=True)
    result = []
    for new_rank, id_ in enumerate(sorted_ids):
        c = chunk_map[id_]
        result.append(Chunk(
            id=c.id,
            doc_id=c.doc_id,
            text=c.text,
            score=scores[id_],
            rank=new_rank,
            embedding=c.embedding,
        ))
    return result


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        if chunks:
            tokenized = [c.text.lower().split() for c in chunks]
            self._bm25: BM25Okapi | None = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def retrieve(self, query_text: str, top_n: int) -> list[Chunk]:
        if not self._chunks or self._bm25 is None:
            return []
        tokens = query_text.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_n]]


# Stub — implemented in Task 8


class HybridRetriever:
    ...
