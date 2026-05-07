from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chroma_persist_path: str | None = None
    run_store_path: str = "runs.db"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_model: str = "qwen2.5:3b"
    default_top_n: int = 20
    default_top_k: int = 5
    default_strong_k: int = 5
    default_token_budget: int = 2048


@lru_cache
def get_settings() -> Settings:
    return Settings()
