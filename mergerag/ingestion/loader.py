from __future__ import annotations

import re
from pathlib import Path


def load_document(path: Path, doc_id: str | None = None) -> tuple[str, str]:
    """Load a document from disk and return (doc_id, raw_text).

    Supported extensions: .txt, .md
    doc_id defaults to path.stem if not provided.
    """
    resolved_id = doc_id if doc_id is not None else path.stem
    ext = path.suffix.lower()

    if ext == ".txt":
        raw_text = path.read_text(encoding="utf-8")
        return resolved_id, raw_text

    if ext == ".md":
        raw_text = path.read_text(encoding="utf-8")
        # Strip fenced code blocks (``` ... ```)
        raw_text = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL)
        # Strip HTML tags
        raw_text = re.sub(r"<[^>]+>", "", raw_text)
        # Normalize excess blank lines
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        return resolved_id, raw_text

    raise ValueError(f"Unsupported file extension: '{ext}'")
