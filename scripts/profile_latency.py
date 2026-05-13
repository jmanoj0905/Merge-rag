"""Profile pipeline stage latency across N questions.

Usage:
    uv run python scripts/profile_latency.py --n 100
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from mergerag.adapters.embedder import SentenceTransformerEmbedder
from mergerag.adapters.llm import OllamaLLM
from mergerag.adapters.retriever import ChromaRetriever
from mergerag.pipeline import MergeRAGPipeline

_STAGE_RE = re.compile(
    r"pipeline stage_ms strategy=(\S+) "
    r"embed=([\d.]+) retrieve=([\d.]+) merge=([\d.]+) "
    r"re_embed=([\d.]+) rerank=([\d.]+) answer=([\d.]+) total=([\d.]+)"
)
_MERGE_OPS_RE = re.compile(r"merge_ops=(\d+)")

STAGES = ["embed", "retrieve", "merge", "re_embed", "rerank", "answer", "total"]


class StageCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []
        self.merge_ops: dict[str, list[int]] = defaultdict(list)

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        m = _STAGE_RE.search(msg)
        if m:
            strategy = m.group(1)
            row = {"strategy": strategy}
            for i, stage in enumerate(STAGES, start=2):
                row[stage] = float(m.group(i))
            self.rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--fixture", default="data/hotpotqa_dev_distractor.json")
    parser.add_argument("--collection", default="hotpot_dev_500")
    parser.add_argument("--persist-path", default="data/chroma")
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()

    capture = StageCapture()
    capture.setLevel(logging.DEBUG)
    pipeline_logger = logging.getLogger("mergerag.pipeline")
    pipeline_logger.setLevel(logging.DEBUG)
    pipeline_logger.addHandler(capture)

    embedder = SentenceTransformerEmbedder()
    retriever = ChromaRetriever(
        collection_name=args.collection, persist_path=args.persist_path
    )
    llm = OllamaLLM(model=args.model)
    pipeline = MergeRAGPipeline(embedder=embedder, retriever=retriever, llm=llm)

    examples = json.loads(Path(args.fixture).read_text())[:args.n]
    print(f"Profiling {args.n} questions across 3 strategies ({args.n * 3} runs)...")

    for i, ex in enumerate(examples):
        for strategy in ("top_k", "symmetric", "asymmetric"):
            pipeline.run(ex["question"], strategy=strategy, collection_name=args.collection)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n} done")

    # Aggregate by strategy
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in capture.rows:
        by_strategy[row["strategy"]].append(row)

    print(f"\n{'Strategy':<12} {'N':>4}  " + "  ".join(f"{s:>10}" for s in STAGES))
    print("-" * (16 + 13 * len(STAGES)))

    for strategy in ("top_k", "symmetric", "asymmetric"):
        rows = by_strategy[strategy]
        n = len(rows)
        if not rows:
            continue
        means = {s: mean(r[s] for r in rows) for s in STAGES}
        line = f"{strategy:<12} {n:>4}  " + "  ".join(f"{means[s]:>10.1f}" for s in STAGES)
        print(line)

    # Asymmetric breakdown detail: how often does merge fire?
    asym_rows = by_strategy["asymmetric"]
    merge_fired = sum(1 for r in asym_rows if r["merge"] > 0)
    re_embed_fired = sum(1 for r in asym_rows if r["re_embed"] > 0)
    print(f"\nAsymmetric merge fired: {merge_fired}/{len(asym_rows)} questions")
    print(f"Asymmetric re_embed fired: {re_embed_fired}/{len(asym_rows)} questions")

    if merge_fired:
        merge_times = [r["merge"] for r in asym_rows if r["merge"] > 0]
        print(f"Asymmetric merge_ms when active: mean={mean(merge_times):.0f} median={median(merge_times):.0f}")

    if re_embed_fired:
        re_times = [r["re_embed"] for r in asym_rows if r["re_embed"] > 0]
        print(f"Asymmetric re_embed_ms when active: mean={mean(re_times):.0f} median={median(re_times):.0f}")

    sym_rows = by_strategy["symmetric"]
    sym_re_embed = sum(1 for r in sym_rows if r["re_embed"] > 0)
    print(f"\nSymmetric re_embed fired: {sym_re_embed}/{len(sym_rows)} questions")


if __name__ == "__main__":
    main()
