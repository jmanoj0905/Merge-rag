"""FastAPI integration tests using TestClient.

Strategy:
- Patch SentenceTransformerEmbedder and OllamaLLM constructors in the
  lifespan so the app starts without needing real models or Ollama.
- After the TestClient context manager enters (lifespan has run), override
  app.state.embedder and app.state.llm with fully-configured MagicMocks.
- The API owns an app-scoped Chroma client, while direct ChromaRetriever
  seeding remains visible through ChromaDB's process-local ephemeral store.
"""
import io
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from mergerag.api.app import app
from mergerag.api.routes.ingest import _read_upload_limited
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
    """TestClient with embedder, LLM, and run store replaced by mocks."""
    with (
        patch("mergerag.api.app.SentenceTransformerEmbedder") as mock_emb_cls,
        patch("mergerag.api.app.OllamaLLM") as mock_llm_cls,
        patch("mergerag.api.app.SQLiteRunStore") as mock_rs_cls,
    ):
        mock_emb_cls.return_value = _make_mock_embedder()
        mock_llm_cls.return_value = _make_mock_llm()
        mock_rs_cls.return_value = MagicMock()

        with TestClient(app) as c:
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

    @pytest.mark.anyio
    async def test_upload_size_limit_returns_413(self):
        upload = UploadFile(filename="large.txt", file=io.BytesIO(b"abcdef"))

        with pytest.raises(HTTPException) as exc:
            await _read_upload_limited(upload, max_bytes=5)

        assert exc.value.status_code == 413


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

    def test_invalid_pipeline_params_return_422(self, client, collection_name):
        resp = client.post(
            "/query",
            json={
                "query": "capitals",
                "strategy": "top_k",
                "collection_name": collection_name,
                "params": {"top_n": 2, "top_k": 3},
            },
        )
        assert resp.status_code == 422

    def test_invalid_pipeline_params_against_defaults_return_422(self, client, collection_name):
        self._seed_collection(collection_name)
        resp = client.post(
            "/query",
            json={
                "query": "capitals",
                "strategy": "top_k",
                "collection_name": collection_name,
                "params": {"strong_k": 100},
            },
        )
        assert resp.status_code == 422

    def test_hybrid_cache_refreshes_after_ingest(self, client, collection_name):
        self._seed_collection(collection_name)
        first = client.post(
            "/query",
            json={
                "query": "capitals",
                "strategy": "top_k",
                "collection_name": collection_name,
                "params": {"retriever": "hybrid"},
            },
        )
        assert first.status_code == 200

        ingest_resp = client.post(
            "/ingest",
            data={"collection_name": collection_name, "doc_id": "zebra_doc"},
            files={"file": ("zebra.txt", io.BytesIO(b"zebra-token appears here."), "text/plain")},
        )
        assert ingest_resp.status_code == 200

        second = client.post(
            "/query",
            json={
                "query": "zebra-token",
                "strategy": "top_k",
                "collection_name": collection_name,
                "params": {"retriever": "hybrid"},
            },
        )
        assert second.status_code == 200
        retrieved_ids = [chunk["doc_id"] for chunk in second.json()["retrieved_chunks"]]
        assert "zebra_doc" in retrieved_ids

    def test_query_runtime_failure_returns_502(self, client, collection_name):
        self._seed_collection(collection_name)
        client.app.state.llm.complete.side_effect = RuntimeError("ollama unavailable")

        resp = client.post(
            "/query",
            json={
                "query": "capitals",
                "strategy": "top_k",
                "collection_name": collection_name,
            },
        )

        assert resp.status_code == 502


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

# ---------------------------------------------------------------------------
# GET /runs  and  GET /runs/{run_id}
# ---------------------------------------------------------------------------

class TestRuns:
    @pytest.fixture
    def runs_client(self, tmp_path):
        from mergerag.adapters.run_store import SQLiteRunStore
        with (
            patch("mergerag.api.app.SentenceTransformerEmbedder") as mock_emb_cls,
            patch("mergerag.api.app.OllamaLLM") as mock_llm_cls,
            patch("mergerag.api.app.SQLiteRunStore") as mock_rs_cls,
        ):
            real_store = SQLiteRunStore(str(tmp_path / "runs.db"))
            mock_rs_cls.return_value = real_store
            mock_emb_cls.return_value = _make_mock_embedder()
            mock_llm_cls.return_value = _make_mock_llm()
            with TestClient(app) as c:
                yield c

    def _seed_and_query(self, client, collection_name: str) -> dict:
        from mergerag.adapters.retriever import ChromaRetriever
        from mergerag.core.models import Chunk
        retriever = ChromaRetriever(collection_name=collection_name)
        chunks = [
            Chunk(id="c1", doc_id="d1", text="Paris is the capital of France.", score=0.0, rank=0),
        ]
        retriever.index(chunks, [_EMBED_VEC])
        resp = client.post("/query", json={
            "query": "capital of France",
            "strategy": "top_k",
            "collection_name": collection_name,
        })
        assert resp.status_code == 200
        return resp.json()

    def test_list_runs_empty_initially(self, runs_client):
        resp = runs_client.get("/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_query_creates_run_record(self, runs_client, collection_name):
        self._seed_and_query(runs_client, collection_name)

        resp = runs_client.get("/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["query"] == "capital of France"
        assert runs[0]["strategy"] == "top_k"
        assert runs[0]["collection_name"] == collection_name
        assert "run_id" in runs[0]
        assert "created_at" in runs[0]
        assert "token_count" in runs[0]
        assert "latency_ms" in runs[0]

    def test_get_run_by_id_returns_full_detail(self, runs_client, collection_name):
        self._seed_and_query(runs_client, collection_name)

        runs = runs_client.get("/runs").json()
        run_id = runs[0]["run_id"]

        resp = runs_client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["run_id"] == run_id
        assert detail["query"] == "capital of France"
        assert detail["answer"] == "Test answer."
        assert "config" in detail
        assert "retrieved_chunks" in detail
        assert "final_context" in detail
        assert "citations" in detail

    def test_get_nonexistent_run_returns_404(self, runs_client):
        resp = runs_client.get("/runs/nonexistent-run-id-xyz")
        assert resp.status_code == 404


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
