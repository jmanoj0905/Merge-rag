from __future__ import annotations

from typing import Any

import chromadb


def make_chroma_client(persist_path: str | None) -> Any:
    if persist_path:
        return chromadb.PersistentClient(path=persist_path)
    return chromadb.EphemeralClient()
