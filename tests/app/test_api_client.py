from unittest.mock import patch, MagicMock
import pytest
import requests as req
from app.api_client import call_query, APIError


def _mock_response(status: int, body: dict) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = body
    mock.text = str(body)
    return mock


_SAMPLE_RESPONSE = {
    "query": "test",
    "strategy": "top_k",
    "answer": "Paris [chunk-0000]",
    "citations": [],
    "token_count": 10,
    "latency_ms": 1234.5,
    "retrieved_chunks": [],
    "merged_chunks": [],
    "final_context": [],
    "merge_plan": None,
}


def test_call_query_returns_response():
    with patch("app.api_client.requests.post", return_value=_mock_response(200, _SAMPLE_RESPONSE)):
        result = call_query("test", "top_k", "col", "http://localhost:8000", timeout=10)
    assert result["answer"] == "Paris [chunk-0000]"
    assert result["latency_ms"] == 1234.5


def test_call_query_raises_on_http_error():
    with patch("app.api_client.requests.post", return_value=_mock_response(500, {})):
        with pytest.raises(APIError, match="500"):
            call_query("test", "top_k", "col", "http://localhost:8000", timeout=10)


def test_call_query_raises_on_connection_error():
    with patch("app.api_client.requests.post", side_effect=req.exceptions.ConnectionError()):
        with pytest.raises(APIError, match="backend not reachable"):
            call_query("test", "top_k", "col", "http://localhost:8000", timeout=10)


def test_call_query_raises_on_timeout():
    with patch("app.api_client.requests.post", side_effect=req.exceptions.Timeout()):
        with pytest.raises(APIError, match="timed out"):
            call_query("test", "top_k", "col", "http://localhost:8000", timeout=10)
