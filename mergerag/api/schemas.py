from typing import Annotated, Literal

from pydantic import BaseModel, Field, PositiveInt, model_validator

from mergerag.core.models import Strategy


# Request models
class PipelineParams(BaseModel):
    top_n: PositiveInt | None = None
    top_k: PositiveInt | None = None
    strong_k: PositiveInt | None = None
    token_budget: PositiveInt | None = None
    asymmetric_max_ops: int | None = Field(default=None, ge=0)
    retriever: Literal["chroma", "hybrid"] | None = None

    @model_validator(mode="after")
    def validate_rank_bounds(self) -> "PipelineParams":
        if self.top_n is not None and self.top_k is not None and self.top_k > self.top_n:
            raise ValueError("top_k must be less than or equal to top_n")
        if self.top_n is not None and self.strong_k is not None and self.strong_k > self.top_n:
            raise ValueError("strong_k must be less than or equal to top_n")
        return self


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


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    query: str
    strategy: Strategy
    collection_name: str
    token_count: int
    latency_ms: float


class RunDetail(BaseModel):
    run_id: str
    created_at: str
    query: str
    strategy: Strategy
    collection_name: str
    config: dict
    answer: str
    citations: list[CitationOut]
    token_count: int
    latency_ms: float
    retrieved_chunks: list[ChunkOut]
    merged_chunks: list[MergedChunkOut]
    final_context: list[ContextItemOut]
    merge_plan: MergePlanOut | None
