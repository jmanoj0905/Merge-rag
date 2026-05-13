from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from mergerag.core.models import RunScore, Strategy
from mergerag.core.ports import (
    EmbedderPort,
    LLMPort,
    RetrieverPort,
    RunStorePort,
    ScoreStorePort,
)
from mergerag.eval.scorer import exact_match, f1
from mergerag.ingestion.ingest import ingest_document
from mergerag.pipeline import MergeRAGPipeline

_BRACKET_RE = re.compile(r"\[([^\]]{1,120})\]")
_CITATION_RE = re.compile(r"\[[^\]]{1,120}\]")
_CHUNK_ID_SUFFIX_RE = re.compile(r"-(?:\d{4,}|[0-9a-f]{8,})$")

logger = logging.getLogger(__name__)


def _strip_citations(text: str) -> str:
    """Remove inline citation tags like [c0] from a pipeline answer."""
    return _CITATION_RE.sub("", text).strip()


def _answer_for_scoring(text: str) -> str:
    """Extract answer text while tolerating common citation-format mistakes."""
    stripped = text.strip()
    without_citations = _strip_citations(stripped)
    if without_citations:
        return without_citations

    bracket_values = [match.strip() for match in _BRACKET_RE.findall(stripped) if match.strip()]
    if not bracket_values:
        return stripped

    cleaned_values: list[str] = []
    for value in bracket_values:
        first_token = re.split(r"[\s,]+", value, maxsplit=1)[0]
        candidate = _CHUNK_ID_SUFFIX_RE.sub("", first_token).replace("_", " ").strip()
        if candidate:
            cleaned_values.append(candidate)
    return " ".join(cleaned_values)


@dataclass
class BenchmarkConfig:
    fixture_path: Path
    collection_name: str
    strategies: list[Strategy] = field(default_factory=lambda: ["top_k", "symmetric", "asymmetric"])
    top_n: int = 20
    top_k: int = 5
    strong_k: int = 5
    token_budget: int = 2048
    limit: int | None = None


@dataclass
class RunResult:
    run_id: str
    question_id: str
    question: str
    strategy: str
    answer: str
    gold_answer: str
    em: float
    f1: float
    latency_ms: float
    token_count: int


@dataclass
class StrategyStats:
    em_mean: float
    f1_mean: float
    latency_ms_mean: float
    token_count_mean: float
    n: int


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    results: list[RunResult]
    summary: dict[str, StrategyStats]
    ran_at: datetime


def _ingest_corpus(
    examples: list[dict],
    embedder: EmbedderPort,
    retriever: RetrieverPort,
) -> None:
    """Ingest all unique document contexts from the fixture examples."""
    ingested_ids: set[str] = set()

    for example in examples:
        for title, sentences in example.get("context", []):
            doc_id = title.replace(" ", "_")
            if doc_id in ingested_ids:
                continue
            ingested_ids.add(doc_id)

            text = "\n\n".join(sentences)
            fd, tmp_path = tempfile.mkstemp(suffix=".txt")
            try:
                os.close(fd)
                Path(tmp_path).write_text(text, encoding="utf-8")
                ingest_document(Path(tmp_path), embedder, retriever, doc_id=doc_id)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _compute_summary(results: list[RunResult], strategies: list[Strategy]) -> dict[str, StrategyStats]:
    summary: dict[str, StrategyStats] = {}
    for strategy in strategies:
        strategy_results = [r for r in results if r.strategy == strategy]
        n = len(strategy_results)
        if n == 0:
            summary[strategy] = StrategyStats(
                em_mean=0.0,
                f1_mean=0.0,
                latency_ms_mean=0.0,
                token_count_mean=0.0,
                n=0,
            )
        else:
            summary[strategy] = StrategyStats(
                em_mean=mean(r.em for r in strategy_results),
                f1_mean=mean(r.f1 for r in strategy_results),
                latency_ms_mean=mean(r.latency_ms for r in strategy_results),
                token_count_mean=mean(r.token_count for r in strategy_results),
                n=n,
            )
    return summary


def run_benchmark(
    config: BenchmarkConfig,
    embedder: EmbedderPort,
    retriever: RetrieverPort,
    llm: LLMPort,
    run_store: RunStorePort,
    score_store: ScoreStorePort,
) -> BenchmarkResult:
    examples: list[dict] = json.loads(Path(config.fixture_path).read_text(encoding="utf-8"))
    if config.limit is not None:
        examples = examples[: config.limit]

    _ingest_corpus(examples, embedder, retriever)

    pipeline = MergeRAGPipeline(
        embedder=embedder,
        retriever=retriever,
        llm=llm,
        top_n=config.top_n,
        top_k=config.top_k,
        strong_k=config.strong_k,
        token_budget=config.token_budget,
    )

    results: list[RunResult] = []
    total = len(examples)

    for idx, example in enumerate(examples):
        question_id = example.get("_id", "")
        question = example.get("question", "")
        gold_answer = example.get("answer", "")

        for strategy in config.strategies:
            try:
                trace = pipeline.run(question, strategy=strategy, collection_name=config.collection_name)
                run_store.save(trace)

                bare_answer = _answer_for_scoring(trace.answer)
                em_score = exact_match(bare_answer, gold_answer)
                f1_score = f1(bare_answer, gold_answer)
                logger.info("[%d/%d] %s em=%.0f lat=%.0fs", idx + 1, total, strategy, em_score, trace.latency_ms / 1000)

                run_score = RunScore(
                    run_id=trace.run_id,
                    question_id=question_id,
                    gold_answer=gold_answer,
                    em=em_score,
                    f1=f1_score,
                )
                score_store.save(run_score)

                results.append(RunResult(
                    run_id=trace.run_id,
                    question_id=question_id,
                    question=question,
                    strategy=strategy,
                    answer=trace.answer,
                    gold_answer=gold_answer,
                    em=em_score,
                    f1=f1_score,
                    latency_ms=trace.latency_ms,
                    token_count=trace.token_count,
                ))

            except Exception:
                logger.warning(
                    "Run failed for question_id=%s strategy=%s",
                    question_id,
                    strategy,
                    exc_info=True,
                )
                results.append(RunResult(
                    run_id="",
                    question_id=question_id,
                    question=question,
                    strategy=strategy,
                    answer="",
                    gold_answer=gold_answer,
                    em=0.0,
                    f1=0.0,
                    latency_ms=0.0,
                    token_count=0,
                ))

    summary = _compute_summary(results, config.strategies)

    return BenchmarkResult(
        config=config,
        results=results,
        summary=summary,
        ran_at=datetime.now(timezone.utc),
    )
