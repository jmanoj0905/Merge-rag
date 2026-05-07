from abc import ABC, abstractmethod
from mergerag.core.models import Chunk


class EmbedderPort(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class RetrieverPort(ABC):
    @abstractmethod
    def retrieve(self, query_embedding: list[float], top_n: int) -> list[Chunk]:
        """Return top_n chunks sorted by descending similarity."""

    @abstractmethod
    def index(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their precomputed embeddings."""


class LLMPort(ABC):
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int) -> str:
        """Return LLM completion for the given prompt."""
