from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mergerag.core.models import Strategy


# Request models
class PipelineParams(BaseModel):
    top_n: int | None = None
    top_k: int | None = None
    strong_k: int | None = None
    token_budget: int | None = None


class QueryRequest(BaseModel):
    query: str
    strategy: Literal["top_k", "symmetric", "asymmetric"]
    collection_name: str
    params: PipelineParams = Field(default_factory=PipelineParams)


# Response models
class ChunkOut(BaseModel):
    type: Literal["chunk"] = "chunk"
    id: str
    doc_id: str
    text: str
    score: float
    rank: int


class MergedChunkOut(BaseModel):
    type: Literal["merged"] = "merged"
    id: str
    text: str
    score: float
    source_chunk_ids: list[str]
    merge_type: Literal["symmetric", "asymmetric"]


class CitationOut(BaseModel):
    sentence: str
    chunk_ids: list[str]


class MergeOpOut(BaseModel):
    type: Literal["symmetric", "asymmetric"]
    primary_id: str
    secondary_id: str


class MergePlanOut(BaseModel):
    operations: list[MergeOpOut]


# Discriminated union for final_context items
ContextItemOut = Annotated[
    ChunkOut | MergedChunkOut,
    Field(discriminator="type")
]


class QueryResponse(BaseModel):
    query: str
    strategy: Strategy
    answer: str
    citations: list[CitationOut]
    token_count: int
    latency_ms: float
    retrieved_chunks: list[ChunkOut]
    merged_chunks: list[MergedChunkOut]
    final_context: list[ContextItemOut]
    merge_plan: MergePlanOut | None


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int
    collection_name: str


class CollectionInfo(BaseModel):
    name: str
    chunk_count: int
