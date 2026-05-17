from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from mergerag.core.models import Chunk
from mergerag.core.ports import EmbedderPort, LLMPort, RetrieverPort
from mergerag.adapters.run_store import SQLiteRunStore
from mergerag.adapters.score_store import SQLiteScoreStore
from mergerag.eval.benchmark import BenchmarkConfig, run_benchmark

_FIXTURE = [
    {
        "_id": "q1",
        "question": "In what country was Python created?",
        "answer": "Netherlands",
        "supporting_facts": [["Python", 1], ["Guido van Rossum", 0]],
        "context": [
            ["Python", ["Python is a programming language.", "Created by Guido van Rossum."]],
            ["Guido van Rossum", ["Guido van Rossum is Dutch.", "Born in Haarlem, Netherlands."]],
        ],
    }
]

_TWO_QUESTION_FIXTURE = [
    *_FIXTURE,
    {
        "_id": "q2",
        "question": "What city is named here?",
        "answer": "Paris",
        "supporting_facts": [["Paris", 0]],
        "context": [
            ["Paris", ["Paris is the capital of France."]],
            ["Berlin", ["Berlin is the capital of Germany."]],
        ],
    },
]


def _write_fixture(data: list) -> Path:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    Path(path).write_text(json.dumps(data))
    return Path(path)


def _mock_embedder() -> EmbedderPort:
    e = MagicMock(spec=EmbedderPort)
    e.embed.side_effect = lambda texts: [[1.0, 0.0]] * len(texts)
    return e


def _mock_retriever() -> RetrieverPort:
    r = MagicMock(spec=RetrieverPort)
    r.retrieve.return_value = [
        Chunk(id=f"c{i}", doc_id="d1", text=f"text {i}", score=1.0 - i * 0.1, rank=i, embedding=[1.0, 0.0])
        for i in range(5)
    ]
    return r


def _mock_retriever_with_one_active_asymmetric_merge() -> RetrieverPort:
    r = MagicMock(spec=RetrieverPort)

    active_chunks = [
        Chunk(id=f"a{i}", doc_id="d1", text=f"active {i}", score=1.0 - i * 0.1, rank=i, embedding=[1.0, 0.0])
        for i in range(5)
    ]
    inactive_chunks = [
        Chunk(id=f"i{i}", doc_id=f"d{i}", text=f"inactive {i}", score=1.0 - i * 0.1, rank=i, embedding=[1.0, 0.0])
        for i in range(5)
    ]

    def retrieve(query, top_n):
        if "country" in query.text:
            return active_chunks
        return inactive_chunks

    r.retrieve.side_effect = retrieve
    return r


def _mock_llm(answer: str = "Netherlands") -> LLMPort:
    llm = MagicMock(spec=LLMPort)
    llm.complete.return_value = f"{answer} [c0]"
    return llm


def _make_stores() -> tuple[SQLiteRunStore, SQLiteScoreStore]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return SQLiteRunStore(path), SQLiteScoreStore(path)


def test_run_benchmark_returns_one_result_per_strategy_per_example():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k", "symmetric"],
    )
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), _mock_llm(), run_store, score_store)
    assert len(result.results) == 2  # 1 example × 2 strategies


def test_run_benchmark_stores_runs_in_run_store():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k"],
    )
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), _mock_llm(), run_store, score_store)
    run_id = result.results[0].run_id
    assert run_store.get(run_id) is not None


def test_run_benchmark_stores_scores_in_score_store():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k"],
    )
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), _mock_llm(), run_store, score_store)
    run_id = result.results[0].run_id
    assert score_store.get(run_id) is not None


def test_run_benchmark_summary_has_all_strategy_keys():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k", "symmetric", "asymmetric"],
    )
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), _mock_llm(), run_store, score_store)
    assert set(result.summary.keys()) == {"top_k", "symmetric", "asymmetric"}


def test_run_benchmark_exact_answer_gives_em_1():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k"],
    )
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), _mock_llm("Netherlands"), run_store, score_store)
    assert result.results[0].em == 1.0


def test_run_benchmark_scores_answer_accidentally_wrapped_in_brackets():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k"],
    )
    llm = MagicMock(spec=LLMPort)
    llm.complete.return_value = "[Netherlands]"
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), llm, run_store, score_store)
    assert result.results[0].em == 1.0


def test_run_benchmark_scores_answer_encoded_like_chunk_id():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k"],
    )
    llm = MagicMock(spec=LLMPort)
    llm.complete.return_value = "[Netherlands-0001]"
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), llm, run_store, score_store)
    assert result.results[0].em == 1.0


def test_run_benchmark_wrong_answer_gives_em_0():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k"],
    )
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), _mock_llm("France"), run_store, score_store)
    assert result.results[0].em == 0.0


def test_run_benchmark_failed_run_records_zero_scores_and_continues():
    fixture_path = _write_fixture(_FIXTURE)
    run_store, score_store = _make_stores()
    llm = MagicMock(spec=LLMPort)
    llm.complete.side_effect = RuntimeError("LLM unavailable")
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k"],
    )
    result = run_benchmark(config, _mock_embedder(), _mock_retriever(), llm, run_store, score_store)
    assert len(result.results) == 1
    assert result.results[0].em == 0.0
    assert result.results[0].f1 == 0.0


def test_run_benchmark_can_keep_only_active_asymmetric_merge_examples():
    fixture_path = _write_fixture(_TWO_QUESTION_FIXTURE)
    run_store, score_store = _make_stores()
    config = BenchmarkConfig(
        fixture_path=fixture_path,
        collection_name="test_col",
        strategies=["top_k", "asymmetric"],
        strong_k=3,
        active_asymmetric_only=True,
    )
    result = run_benchmark(
        config,
        _mock_embedder(),
        _mock_retriever_with_one_active_asymmetric_merge(),
        _mock_llm(),
        run_store,
        score_store,
    )
    assert len(result.results) == 2
    assert {r.question_id for r in result.results} == {"q1"}
