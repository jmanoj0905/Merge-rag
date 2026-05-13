from __future__ import annotations
import logging
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
logger = logging.getLogger(__name__)


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
        collection_name: str = "",
    ) -> RunTrace:
        t0 = time.time()

        t_embed_start = time.time()
        query_emb = self._embedder.embed([query])[0]
        t_embed = time.time() - t_embed_start

        t_retrieve_start = time.time()
        chunks = self._retriever.retrieve(query_emb, self._top_n)
        t_retrieve = time.time() - t_retrieve_start

        merge_plan: MergePlan | None = None
        merged: list[MergedChunk] = []
        t_merge = t_re_embed = t_rerank = 0.0

        if strategy == "top_k":
            final_context: list[Chunk | MergedChunk] = chunks[: self._top_k]
        else:
            merge_plan = planner.plan(chunks, strategy, self._strong_k)

            t_merge_start = time.time()
            merged = executor.execute(merge_plan, query, self._llm)
            t_merge = time.time() - t_merge_start

            if merged:
                t_re_embed_start = time.time()
                merged_embs = self._embedder.embed([m.text for m in merged])
                t_re_embed = time.time() - t_re_embed_start
                for m, emb in zip(merged, merged_embs):
                    m.embedding = emb

            t_rerank_start = time.time()
            pool: list[Chunk | MergedChunk] = list(chunks[: self._strong_k]) + merged
            pool = [x for x in pool if x.embedding]
            pool.sort(key=lambda x: cosine_similarity(x.embedding, query_emb), reverse=True)
            final_context = pool[: self._top_k]
            t_rerank = time.time() - t_rerank_start

        t_answer_start = time.time()
        answer, citations = _generate_answer(final_context, query, self._llm)
        t_answer = time.time() - t_answer_start

        latency_ms = (time.time() - t0) * 1000

        logger.debug(
            "pipeline stage_ms strategy=%s embed=%.0f retrieve=%.0f merge=%.0f "
            "re_embed=%.0f rerank=%.0f answer=%.0f total=%.0f",
            strategy,
            t_embed * 1000, t_retrieve * 1000, t_merge * 1000,
            t_re_embed * 1000, t_rerank * 1000, t_answer * 1000,
            latency_ms,
        )

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
            collection_name=collection_name,
            config={
                "top_n": self._top_n,
                "top_k": self._top_k,
                "strong_k": self._strong_k,
                "token_budget": self._token_budget,
            },
        )
