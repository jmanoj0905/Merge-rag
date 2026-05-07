from __future__ import annotations

from mergerag.core.models import Chunk


class ParagraphChunker:
    def __init__(self, max_chars: int = 1000, min_chars: int = 100):
        if max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {max_chars}")
        if min_chars < 0:
            raise ValueError(f"min_chars must be non-negative, got {min_chars}")
        if min_chars >= max_chars:
            raise ValueError(f"min_chars ({min_chars}) must be less than max_chars ({max_chars})")
        self._max_chars = max_chars
        self._min_chars = min_chars

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        # Step 1: split on double newlines, strip, drop empties
        paragraphs = [p.strip() for p in text.split("\n\n")]
        paragraphs = [p for p in paragraphs if p]

        if not paragraphs:
            return []

        # Step 2: greedy merge
        pieces: list[str] = []
        current = paragraphs[0]

        for para in paragraphs[1:]:
            # Would adding this paragraph (with separator) exceed max_chars?
            candidate = current + "\n\n" + para
            if len(candidate) > self._max_chars:
                pieces.append(current)
                current = para
            else:
                current = candidate

        # Step 3: handle the last piece
        if pieces and len(current) < self._min_chars:
            # Fold into the previous chunk
            pieces[-1] = pieces[-1] + "\n\n" + current
        else:
            pieces.append(current)

        # Step 4: build Chunk objects with deterministic IDs
        return [
            Chunk(
                id=f"{doc_id}-{index:04d}",
                doc_id=doc_id,
                text=piece,
                score=0.0,
                rank=0,
                embedding=[],
            )
            for index, piece in enumerate(pieces)
        ]
