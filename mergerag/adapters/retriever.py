import chromadb
from mergerag.core.models import Chunk
from mergerag.core.ports import RetrieverPort


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

    def retrieve(self, query_embedding: list[float], top_n: int) -> list[Chunk]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
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
