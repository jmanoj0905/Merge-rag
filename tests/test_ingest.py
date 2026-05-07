from pathlib import Path

from mergerag.adapters.embedder import SentenceTransformerEmbedder
from mergerag.adapters.retriever import ChromaRetriever
from mergerag.ingestion.chunker import ParagraphChunker
from mergerag.ingestion.ingest import ingest_document


def _make_three_paragraph_doc(tmp_path: Path) -> Path:
    content = (
        "First paragraph. This is the opening section of the document "
        "and provides an introduction to the topic at hand.\n\n"
        "Second paragraph. This is the middle section of the document "
        "and expands on the ideas introduced in the first paragraph.\n\n"
        "Third paragraph. This is the closing section and wraps up "
        "the document with a conclusion."
    )
    doc = tmp_path / "sample.txt"
    doc.write_text(content, encoding="utf-8")
    return doc


def test_ingest_document_returns_chunk_count(tmp_path: Path):
    doc = _make_three_paragraph_doc(tmp_path)
    embedder = SentenceTransformerEmbedder()
    retriever = ChromaRetriever(collection_name="test_ingest_count")
    chunker = ParagraphChunker(max_chars=1000, min_chars=10)

    count = ingest_document(doc, embedder, retriever, chunker=chunker)

    # With max_chars=1000 all three paragraphs fit in one chunk (each is short)
    # but we're checking the return value matches actual chunk production
    expected_chunks = chunker.chunk("sample", doc.read_text(encoding="utf-8"))
    assert count == len(expected_chunks)


def test_ingest_document_indexes_chunks_in_chromadb(tmp_path: Path):
    doc = _make_three_paragraph_doc(tmp_path)
    embedder = SentenceTransformerEmbedder()
    retriever = ChromaRetriever(collection_name="test_ingest_chroma")
    chunker = ParagraphChunker(max_chars=200, min_chars=10)

    count = ingest_document(doc, embedder, retriever, chunker=chunker)

    # Query ChromaDB directly via the internal collection
    chroma_count = retriever._collection.count()
    assert chroma_count == count
    assert count > 0


def test_ingest_document_respects_doc_id_override(tmp_path: Path):
    doc = _make_three_paragraph_doc(tmp_path)
    embedder = SentenceTransformerEmbedder()
    retriever = ChromaRetriever(collection_name="test_ingest_docid")

    ingest_document(doc, embedder, retriever, doc_id="override-id")

    results = retriever._collection.get(where={"doc_id": "override-id"})
    assert len(results["ids"]) > 0
