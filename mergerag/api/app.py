import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from mergerag.adapters.embedder import SentenceTransformerEmbedder
from mergerag.adapters.llm import OllamaLLM
from mergerag.adapters.run_store import SQLiteRunStore
from mergerag.api.chroma import make_chroma_client
from mergerag.api.config import get_settings
from mergerag.api.routes import query as query_router
from mergerag.api.routes import ingest as ingest_router
from mergerag.api.routes import collections as collections_router
from mergerag.api.routes import runs as runs_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading embedder: %s", settings.embedding_model)
    app.state.embedder = SentenceTransformerEmbedder(settings.embedding_model)
    logger.info("Loading LLM: %s", settings.ollama_model)
    app.state.llm = OllamaLLM(settings.ollama_model)
    logger.info("Initialising run store: %s", settings.run_store_path)
    app.state.run_store = SQLiteRunStore(settings.run_store_path)
    app.state.chroma_client = make_chroma_client(settings.chroma_persist_path)
    logger.info("Startup complete")
    yield


app = FastAPI(title="MergeRAG", version="0.1.0", lifespan=lifespan)

app.include_router(query_router.router)
app.include_router(ingest_router.router)
app.include_router(collections_router.router)
app.include_router(runs_router.router)
