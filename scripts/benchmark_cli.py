from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from mergerag.adapters.embedder import SentenceTransformerEmbedder
from mergerag.adapters.llm import OllamaLLM
from mergerag.adapters.retriever import ChromaRetriever, HybridRetriever
from mergerag.adapters.run_store import SQLiteRunStore
from mergerag.adapters.score_store import SQLiteScoreStore
from mergerag.eval.benchmark import BenchmarkConfig, run_benchmark

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _reset_collection(name: str, persist_path: str | None) -> None:
    if persist_path is None:
        return
    client = chromadb.PersistentClient(path=persist_path)
    try:
        client.delete_collection(name)
    except Exception:
        pass


def _print_summary(summary: dict) -> None:
    header = f"\n{'strategy':<14} {'EM':>6} {'F1':>6} {'latency_ms':>12} {'tokens':>8}"
    print(header)
    print("-" * len(header))
    for strategy, stats in summary.items():
        print(
            f"{strategy:<14} {stats.em_mean:>6.2f} {stats.f1_mean:>6.2f} "
            f"{stats.latency_ms_mean:>12.0f} {stats.token_count_mean:>8.0f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MergeRAG benchmark")
    parser.add_argument("--fixture", required=True, type=Path, help="Path to HotpotQA JSON fixture")
    parser.add_argument("--collection", required=True, help="Chroma collection name")
    parser.add_argument("--db", default="runs.db", help="SQLite DB path for runs and scores")
    parser.add_argument("--output", type=Path, default=None, help="JSON results file path")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["top_k", "symmetric", "asymmetric"],
        default=["top_k", "symmetric", "asymmetric"],
    )
    parser.add_argument("--persist-path", default=None, help="Chroma persistence directory (omit for ephemeral)")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of examples (e.g. 500 or 1000)")
    parser.add_argument("--retriever", choices=["chroma", "hybrid"], default="chroma", help="Retriever backend (default: chroma)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output: Path = args.output or Path(f"benchmark_{timestamp}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    _reset_collection(args.collection, args.persist_path)

    embedder = SentenceTransformerEmbedder()
    if args.retriever == "hybrid":
        retriever = HybridRetriever(collection_name=args.collection, persist_path=args.persist_path)
    else:
        retriever = ChromaRetriever(collection_name=args.collection, persist_path=args.persist_path)
    llm = OllamaLLM()
    run_store = SQLiteRunStore(db_path=args.db)
    score_store = SQLiteScoreStore(db_path=args.db)

    config = BenchmarkConfig(
        fixture_path=args.fixture,
        collection_name=args.collection,
        strategies=args.strategies,
        limit=args.limit,
    )

    result = run_benchmark(config, embedder, retriever, llm, run_store, score_store)

    _print_summary(result.summary)

    payload = {
        "ran_at": result.ran_at.isoformat(),
        "fixture": str(config.fixture_path),
        "collection": config.collection_name,
        "summary": {
            s: {
                "em_mean": stats.em_mean,
                "f1_mean": stats.f1_mean,
                "latency_ms_mean": stats.latency_ms_mean,
                "token_count_mean": stats.token_count_mean,
                "n": stats.n,
            }
            for s, stats in result.summary.items()
        },
        "results": [
            {
                "run_id": r.run_id,
                "question_id": r.question_id,
                "question": r.question,
                "strategy": r.strategy,
                "answer": r.answer,
                "gold_answer": r.gold_answer,
                "em": r.em,
                "f1": r.f1,
                "latency_ms": r.latency_ms,
                "token_count": r.token_count,
            }
            for r in result.results
        ],
    }
    output.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
