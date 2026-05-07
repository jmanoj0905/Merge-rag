import pytest
from pathlib import Path

from mergerag.ingestion.loader import load_document


def test_txt_content_returned_as_is(tmp_path: Path):
    doc = tmp_path / "sample.txt"
    doc.write_text("Hello world.\nSecond line.", encoding="utf-8")
    doc_id, text = load_document(doc)
    assert doc_id == "sample"
    assert text == "Hello world.\nSecond line."


def test_txt_doc_id_defaults_to_stem(tmp_path: Path):
    doc = tmp_path / "my_doc.txt"
    doc.write_text("content", encoding="utf-8")
    doc_id, _ = load_document(doc)
    assert doc_id == "my_doc"


def test_md_strips_fenced_code_blocks(tmp_path: Path):
    content = "Intro prose.\n\n```python\nx = 1\n```\n\nOutro prose."
    doc = tmp_path / "guide.md"
    doc.write_text(content, encoding="utf-8")
    _, text = load_document(doc)
    assert "x = 1" not in text
    assert "Intro prose." in text
    assert "Outro prose." in text


def test_md_strips_html_tags(tmp_path: Path):
    content = "Some <b>bold</b> text and <br/> a break."
    doc = tmp_path / "page.md"
    doc.write_text(content, encoding="utf-8")
    _, text = load_document(doc)
    assert "<b>" not in text
    assert "<br/>" not in text
    assert "bold" in text


def test_md_preserves_prose(tmp_path: Path):
    content = "First paragraph.\n\nSecond paragraph."
    doc = tmp_path / "notes.md"
    doc.write_text(content, encoding="utf-8")
    _, text = load_document(doc)
    assert "First paragraph." in text
    assert "Second paragraph." in text


def test_doc_id_override(tmp_path: Path):
    doc = tmp_path / "original.txt"
    doc.write_text("content", encoding="utf-8")
    doc_id, _ = load_document(doc, doc_id="custom-id")
    assert doc_id == "custom-id"


def test_unsupported_extension_raises_value_error(tmp_path: Path):
    doc = tmp_path / "file.pdf"
    doc.write_text("%PDF-1.4", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.pdf"):
        load_document(doc)
