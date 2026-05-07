import pytest

from mergerag.ingestion.chunker import ParagraphChunker


def test_single_paragraph_under_max_chars():
    chunker = ParagraphChunker(max_chars=1000, min_chars=10)
    chunks = chunker.chunk("doc", "Short paragraph.")
    assert len(chunks) == 1
    assert chunks[0].text == "Short paragraph."


def test_two_paragraphs_exceeding_max_chars_become_two_chunks():
    chunker = ParagraphChunker(max_chars=30, min_chars=5)
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = chunker.chunk("doc", text)
    assert len(chunks) == 2
    assert "First" in chunks[0].text
    assert "Second" in chunks[1].text


def test_short_trailing_paragraph_folded_into_previous():
    # para1+para2 together (80+2+80=162 chars) exceed max_chars=100,
    # so they split into two chunks. para3 is tiny (< min_chars=50)
    # and gets folded into the second chunk.
    chunker = ParagraphChunker(max_chars=100, min_chars=50)
    para1 = "A" * 80
    para2 = "B" * 80
    para3 = "tiny"  # 4 chars < min_chars=50
    text = f"{para1}\n\n{para2}\n\n{para3}"
    chunks = chunker.chunk("doc", text)
    assert len(chunks) == 2
    assert "tiny" in chunks[-1].text
    assert "B" * 80 in chunks[-1].text


def test_very_long_single_paragraph_emitted_as_one_chunk():
    chunker = ParagraphChunker(max_chars=50, min_chars=10)
    long_para = "X" * 500
    chunks = chunker.chunk("doc", long_para)
    assert len(chunks) == 1
    assert chunks[0].text == long_para


def test_chunk_ids_are_deterministic():
    chunker = ParagraphChunker(max_chars=30, min_chars=5)
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = chunker.chunk("mydoc", text)
    assert chunks[0].id == "mydoc-0000"
    assert chunks[1].id == "mydoc-0001"


def test_chunk_defaults():
    chunker = ParagraphChunker()
    chunks = chunker.chunk("doc", "Some text here.")
    for chunk in chunks:
        assert chunk.score == 0.0
        assert chunk.rank == 0
        assert chunk.embedding == []


def test_chunk_doc_id_set_correctly():
    chunker = ParagraphChunker()
    chunks = chunker.chunk("my-document", "Some text.")
    assert chunks[0].doc_id == "my-document"


def test_invalid_config_min_chars_exceeds_max_chars():
    with pytest.raises(ValueError, match="min_chars .* must be less than max_chars"):
        ParagraphChunker(max_chars=50, min_chars=100)
