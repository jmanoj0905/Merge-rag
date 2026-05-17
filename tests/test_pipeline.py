from unittest.mock import MagicMock
from mergerag.core.models import Chunk, MergedChunk
from mergerag.core.ports import EmbedderPort, RetrieverPort, LLMPort
from mergerag.pipeline import MergeRAGPipeline


def _make_chunk(id_: str, rank: int) -> Chunk:
    return Chunk(
        id=id_, doc_id="d1", text=f"text {id_}",
        score=1.0 - rank * 0.05, rank=rank,
        embedding=[1.0, 0.0],
    )


def _mock_embedder() -> EmbedderPort:
    e = MagicMock(spec=EmbedderPort)
    e.embed.side_effect = lambda texts: [[1.0, 0.0]] * len(texts)
    return e


def _mock_retriever(n: int = 10) -> RetrieverPort:
    r = MagicMock(spec=RetrieverPort)
    r.retrieve.return_value = [_make_chunk(f"c{i}", i) for i in range(n)]
    return r


def _mock_llm() -> LLMPort:
    llm = MagicMock(spec=LLMPort)
    llm.complete.return_value = "The answer is X. [c0]"
    return llm


def test_top_k_run_returns_trace():
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(),
        llm=_mock_llm(),
        top_n=10, top_k=3, strong_k=3,
    )
    trace = pipeline.run("what is X?", strategy="top_k")
    assert trace.strategy == "top_k"
    assert trace.merge_plan is None
    assert len(trace.final_context) == 3
    assert trace.answer != ""


def test_symmetric_run_returns_trace_with_merges():
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(n=10),
        llm=_mock_llm(),
        top_n=10, top_k=3, strong_k=3,
    )
    trace = pipeline.run("what is X?", strategy="symmetric")
    assert trace.strategy == "symmetric"
    assert trace.merge_plan is not None
    assert len(trace.merge_plan.operations) > 0


def test_asymmetric_run_produces_merged_chunks():
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(n=10),
        llm=_mock_llm(),
        top_n=10, top_k=3, strong_k=3,
    )
    trace = pipeline.run("what is X?", strategy="asymmetric")
    assert trace.strategy == "asymmetric"
    assert len(trace.merged_chunks) > 0


def test_asymmetric_run_defaults_to_one_merge_llm_call():
    llm = _mock_llm()
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(n=10),
        llm=llm,
        top_n=10, top_k=3, strong_k=3,
    )
    trace = pipeline.run("what is X?", strategy="asymmetric")
    assert len(trace.merge_plan.operations) == 1
    assert trace.config["asymmetric_max_ops"] == 1
    assert llm.complete.call_count == 2  # one merge synthesis + one final answer


def test_asymmetric_run_can_raise_merge_llm_call_cap():
    llm = _mock_llm()
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(n=10),
        llm=llm,
        top_n=10, top_k=3, strong_k=3,
        asymmetric_max_ops=3,
    )
    trace = pipeline.run("what is X?", strategy="asymmetric")
    assert len(trace.merge_plan.operations) == 3
    assert llm.complete.call_count == 4  # three merge syntheses + one final answer


def test_run_trace_has_latency():
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(),
        llm=_mock_llm(),
    )
    trace = pipeline.run("q", strategy="top_k")
    assert trace.latency_ms > 0


def test_run_trace_token_count_is_nonzero():
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(),
        llm=_mock_llm(),
        top_k=3,
    )
    trace = pipeline.run("q", strategy="top_k")
    assert trace.token_count > 0


def test_run_populates_collection_name():
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(),
        llm=_mock_llm(),
    )
    trace = pipeline.run("q", strategy="top_k", collection_name="my_col")
    assert trace.collection_name == "my_col"


def test_run_populates_config_with_pipeline_params():
    pipeline = MergeRAGPipeline(
        embedder=_mock_embedder(),
        retriever=_mock_retriever(),
        llm=_mock_llm(),
        top_n=15, top_k=4, strong_k=3, token_budget=1024,
    )
    trace = pipeline.run("q", strategy="top_k")
    assert trace.config["top_n"] == 15
    assert trace.config["top_k"] == 4
    assert trace.config["strong_k"] == 3
    assert trace.config["token_budget"] == 1024
    assert trace.config["asymmetric_max_ops"] == 1
