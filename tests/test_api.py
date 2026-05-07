"""FastAPI integration tests using TestClient.

Strategy:
- Patch SentenceTransformerEmbedder and OllamaLLM constructors in the
  lifespan so the app starts without needing real models or Ollama.
- After the TestClient context manager enters (lifespan has run), override
  app.state.embedder and app.state.llm with fully-configured MagicMocks.
- ChromaDB's EphemeralClient is shared within a process, so collections
  created via ChromaRetriever in tests are visible to the route's own client.
"""
import io
import uuid

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from mergerag.api.app import app
from mergerag.adapters.retriever import ChromaRetriever

_EMBED_DIM = 384
_EMBED_VEC = [0.1] * _EMBED_DIM


def _make_mock_embedder() -> MagicMock:
    mock = MagicMock()
    mock.embed.return_value = [_EMBED_VEC]
    return mock


def _make_mock_llm() -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = "Test answer."
    return mock


@pytest.fixture
def client():
    """TestClient with embedder and LLM replaced by mocks."""
    with (
        patch("mergerag.api.app.SentenceTransformerEmbedder") as mock_emb_cls,
        patch("mergerag.api.app.OllamaLLM") as mock_llm_cls,
    ):
        mock_emb_cls.return_value = _make_mock_embedder()
        mock_llm_cls.return_value = _make_mock_llm()

        with TestClient(app) as c:
            # app.state.embedder/llm are already mock_emb_cls.return_value /
            # mock_llm_cls.return_value after lifespan runs — no override needed.
            yield c


@pytest.fixture
def collection_name() -> str:
    """Return a unique collection name per test to avoid cross-test pollution."""
    return f"test_col_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_happy_path_txt(self, client, collection_name):
        content = b"First paragraph of the document.\n\nSecond paragraph here."
        resp = client.post(
            "/ingest",
            data={"collection_name": collection_name},
            files={"file": ("sample.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == "sample"
        assert body["chunk_count"] >= 1
        assert body["collection_name"] == collection_name

    def test_happy_path_md(self, client, collection_name):
        content = b"# Heading\n\nSome markdown content."
        resp = client.post(
            "/ingest",
            data={"collection_name": collection_name},
            files={"file": ("doc.md", io.BytesIO(content), "text/markdown")},
        )
        assert resp.status_code == 200

    def test_unsupported_extension_returns_422(self, client, collection_name):
        content = b"%PDF-1.4 fake pdf"
        resp = client.post(
            "/ingest",
            data={"collection_name": collection_name},
            files={"file": ("report.pdf", io.BytesIO(content), "application/pdf")},
        )
        assert resp.status_code == 422

    def test_explicit_doc_id(self, client, collection_name):
        content = b"Some text content."
        resp = client.post(
            "/ingest",
            data={"collection_name": collection_name, "doc_id": "my_doc"},
            files={"file": ("anything.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == "my_doc"


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class TestQuery:
    def _seed_collection(self, collection_name: str) -> None:
        """Index a couple of chunks directly via ChromaRetriever."""
        from mergerag.core.models import Chunk

        retriever = ChromaRetriever(collection_name=collection_name)
        chunks = [
            Chunk(id="c1", doc_id="doc1", text="Paris is the capital of France.", score=0.0, rank=0),
            Chunk(id="c2", doc_id="doc1", text="Berlin is the capital of Germany.", score=0.0, rank=1),
        ]
        embeddings = [_EMBED_VEC, _EMBED_VEC]
        retriever.index(chunks, embeddings)

    def test_happy_path_top_k(self, client, collection_name):
        self._seed_collection(collection_name)

        resp = client.post(
            "/query",
            json={
                "query": "What is the capital of France?",
                "strategy": "top_k",
                "collection_name": collection_name,
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        expected_keys = {
            "query", "strategy", "answer", "citations",
            "token_count", "latency_ms", "retrieved_chunks",
            "merged_chunks", "final_context",
        }
        assert expected_keys.issubset(body.keys())
        assert body["answer"] == "Test answer."
        assert body["query"] == "What is the capital of France?"
        assert body["strategy"] == "top_k"

    def test_happy_path_via_ingest_then_query(self, client, collection_name):
        """Seed via /ingest, then query — validates the full round-trip."""
        content = b"The Eiffel Tower is in Paris.\n\nParis is the capital of France."
        ingest_resp = client.post(
            "/ingest",
            data={"collection_name": collection_name},
            files={"file": ("paris.txt", io.BytesIO(content), "text/plain")},
        )
        assert ingest_resp.status_code == 200

        query_resp = client.post(
            "/query",
            json={
                "query": "Where is the Eiffel Tower?",
                "strategy": "top_k",
                "collection_name": collection_name,
            },
        )
        assert query_resp.status_code == 200
        assert query_resp.json()["answer"] == "Test answer."

    def test_unknown_collection_returns_404(self, client):
        resp = client.post(
            "/query",
            json={
                "query": "anything",
                "strategy": "top_k",
                "collection_name": "nonexistent_collection_xyz_123",
            },
        )
        assert resp.status_code == 404

    def test_response_lists_are_correct_types(self, client, collection_name):
        self._seed_collection(collection_name)

        resp = client.post(
            "/query",
            json={
                "query": "capitals",
                "strategy": "top_k",
                "collection_name": collection_name,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["retrieved_chunks"], list)
        assert isinstance(body["merged_chunks"], list)
        assert isinstance(body["final_context"], list)
        assert isinstance(body["citations"], list)


# ---------------------------------------------------------------------------
# GET /collections
# ---------------------------------------------------------------------------

class TestListCollections:
    def test_returns_200_and_list(self, client):
        resp = client.get("/collections")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_newly_created_collection_appears(self, client, collection_name):
        # Create a collection by ingesting
        content = b"Some content."
        client.post(
            "/ingest",
            data={"collection_name": collection_name},
            files={"file": ("doc.txt", io.BytesIO(content), "text/plain")},
        )
        resp = client.get("/collections")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.json()]
        assert collection_name in names


# ---------------------------------------------------------------------------
# DELETE /collections/{name}
# ---------------------------------------------------------------------------

class TestDeleteCollection:
    def test_nonexistent_collection_returns_404(self, client):
        resp = client.delete("/collections/nonexistent_collection_xyz_456")
        assert resp.status_code == 404

    def test_delete_existing_collection(self, client, collection_name):
        # Create the collection first
        content = b"Content to delete."
        ingest_resp = client.post(
            "/ingest",
            data={"collection_name": collection_name},
            files={"file": ("del.txt", io.BytesIO(content), "text/plain")},
        )
        assert ingest_resp.status_code == 200

        resp = client.delete(f"/collections/{collection_name}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == collection_name
