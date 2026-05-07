from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

Strategy = Literal["top_k", "symmetric", "asymmetric"]


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    score: float
    rank: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class MergedChunk:
    id: str
    text: str
    score: float
    source_chunk_ids: list[str]
    merge_type: Literal["symmetric", "asymmetric"]
    embedding: list[float] = field(default_factory=list)


@dataclass
class MergeOp:
    type: Literal["symmetric", "asymmetric"]
    primary: Chunk
    secondary: Chunk


@dataclass
class MergePlan:
    operations: list[MergeOp]


@dataclass
class Citation:
    sentence: str
    chunk_ids: list[str]


@dataclass
class RunTrace:
    query: str
    strategy: Literal["top_k", "symmetric", "asymmetric"]
    retrieved_chunks: list[Chunk]
    merge_plan: MergePlan | None
    merged_chunks: list[MergedChunk]
    final_context: list[Chunk | MergedChunk]
    answer: str
    citations: list[Citation]
    token_count: int
    latency_ms: float
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    collection_name: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class RunScore:
    run_id: str
    question_id: str
    gold_answer: str
    em: float
    f1: float
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
