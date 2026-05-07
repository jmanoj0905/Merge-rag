from __future__ import annotations

from pathlib import Path

from mergerag.core.ports import EmbedderPort, RetrieverPort
from mergerag.ingestion.chunker import ParagraphChunker
from mergerag.ingestion.loader import load_document


def ingest_document(
    path: Path,
    embedder: EmbedderPort,
    retriever: RetrieverPort,
    chunker: ParagraphChunker | None = None,
    doc_id: str | None = None,
) -> int:
    """Load, chunk, embed, and index a document. Returns the number of chunks indexed."""
    if chunker is None:
        chunker = ParagraphChunker()

    resolved_id, raw_text = load_document(path, doc_id=doc_id)
    chunks = chunker.chunk(resolved_id, raw_text)

    if not chunks:
        return 0

    embeddings = embedder.embed([c.text for c in chunks])
    retriever.index(chunks, embeddings)
    return len(chunks)
