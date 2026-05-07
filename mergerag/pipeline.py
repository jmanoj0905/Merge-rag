from __future__ import annotations
import re
import time
from pathlib import Path

from mergerag.core.models import (
    Chunk, MergedChunk, MergePlan, Citation, RunTrace, Strategy,
)
from mergerag.core.ports import EmbedderPort, RetrieverPort, LLMPort
from mergerag.core.utils import cosine_similarity
from mergerag.merge import planner, executor

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _count_tokens(items: list[Chunk | MergedChunk]) -> int:
    return sum(len(item.text.split()) for item in items)


def _parse_citations(answer: str) -> list[Citation]:
    citations: list[Citation] = []
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    for sentence in sentences:
        ids = [
            cid.strip()
            for match in re.findall(r"\[([^\]]+)\]", sentence)
            for cid in match.split(",")
        ]
        if ids:
            citations.append(Citation(sentence=sentence, chunk_ids=ids))
    return citations


def _generate_answer(
    context: list[Chunk | MergedChunk],
    query: str,
    llm: LLMPort,
) -> tuple[str, list[Citation]]:
    template = (_PROMPTS_DIR / "answer.txt").read_text()
    context_text = "\n\n".join(f"[{item.id}]\n{item.text}" for item in context)
    prompt = template.format(query=query, context=context_text)
    answer = llm.complete(prompt, max_tokens=1024)
    return answer, _parse_citations(answer)


class MergeRAGPipeline:
    def __init__(
        self,
        embedder: EmbedderPort,
        retriever: RetrieverPort,
        llm: LLMPort,
        top_n: int = 20,
        top_k: int = 5,
        strong_k: int = 5,
        token_budget: int = 2048,
    ):
        self._embedder = embedder
        self._retriever = retriever
        self._llm = llm
        self._top_n = top_n
        self._top_k = top_k
        self._strong_k = strong_k
        self._token_budget = token_budget

    def run(
        self,
        query: str,
        strategy: Strategy,
    ) -> RunTrace:
        t0 = time.time()

        query_emb = self._embedder.embed([query])[0]
        chunks = self._retriever.retrieve(query_emb, self._top_n)

        merge_plan: MergePlan | None = None
        merged: list[MergedChunk] = []

        if strategy == "top_k":
            final_context: list[Chunk | MergedChunk] = chunks[: self._top_k]
        else:
            merge_plan = planner.plan(chunks, strategy, self._strong_k)
            merged = executor.execute(merge_plan, query, self._llm)

            if merged:
                merged_embs = self._embedder.embed([m.text for m in merged])
                for m, emb in zip(merged, merged_embs):
                    m.embedding = emb

            pool: list[Chunk | MergedChunk] = list(chunks[: self._strong_k]) + merged
            pool = [x for x in pool if x.embedding]
            pool.sort(key=lambda x: cosine_similarity(x.embedding, query_emb), reverse=True)
            final_context = pool[: self._top_k]

        answer, citations = _generate_answer(final_context, query, self._llm)
        latency_ms = (time.time() - t0) * 1000

        return RunTrace(
            query=query,
            strategy=strategy,
            retrieved_chunks=chunks,
            merge_plan=merge_plan,
            merged_chunks=merged,
            final_context=final_context,
            answer=answer,
            citations=citations,
            token_count=_count_tokens(final_context),
            latency_ms=latency_ms,
        )
