from __future__ import annotations
import requests


class APIError(Exception):
    pass


def call_query(
    query: str,
    strategy: str,
    collection_name: str,
    base_url: str,
    timeout: int = 120,
    retriever: str | None = None,
) -> dict:
    """POST /query and return the parsed JSON response dict."""
    params: dict = {}
    if retriever:
        params["retriever"] = retriever
    try:
        resp = requests.post(
            f"{base_url}/query",
            json={
                "query": query,
                "strategy": strategy,
                "collection_name": collection_name,
                "params": params,
            },
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        raise APIError(
            "backend not reachable — run: uv run uvicorn mergerag.api.app:app"
        )
    except requests.exceptions.Timeout:
        raise APIError(f"request timed out after {timeout}s")

    if resp.status_code != 200:
        raise APIError(f"{resp.status_code}: {resp.text[:200]}")

    return resp.json()
