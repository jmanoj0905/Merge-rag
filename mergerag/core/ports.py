from abc import ABC, abstractmethod
from mergerag.core.models import Chunk, Query, RunScore, RunTrace


class EmbedderPort(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class RetrieverPort(ABC):
    @abstractmethod
    def retrieve(self, query: Query, top_n: int) -> list[Chunk]:
        """Return top_n chunks sorted by descending relevance score."""

    @abstractmethod
    def index(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their precomputed embeddings."""


class LLMPort(ABC):
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int) -> str:
        """Return LLM completion for the given prompt."""


class RunStorePort(ABC):
    @abstractmethod
    def save(self, run: RunTrace) -> None: ...

    @abstractmethod
    def get(self, run_id: str) -> RunTrace | None: ...

    @abstractmethod
    def list_runs(self, limit: int = 50, offset: int = 0) -> list[RunTrace]: ...


class ScoreStorePort(ABC):
    @abstractmethod
    def save(self, score: RunScore) -> None: ...

    @abstractmethod
    def get(self, run_id: str) -> RunScore | None: ...

    @abstractmethod
    def list_scores(self, limit: int = 100, offset: int = 0) -> list[RunScore]: ...
