from __future__ import annotations
import re

_CITATION_RE = re.compile(r"\[[^\]]{1,120}\]")


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def compute_em(answer: str, gold: str) -> float:
    return float(strip_citations(answer).lower().strip() == gold.lower().strip())
