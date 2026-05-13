from unittest.mock import MagicMock
from mergerag.core.models import Chunk, MergeOp, MergePlan
from mergerag.core.ports import LLMPort
from mergerag.merge.executor import execute


def _chunk(id_: str) -> Chunk:
    return Chunk(id=id_, doc_id="d1", text=f"text of {id_}", score=0.5, rank=0)


def _mock_llm(response: str = "merged text") -> LLMPort:
    llm = MagicMock(spec=LLMPort)
    llm.complete.return_value = response
    return llm


def test_execute_returns_one_merged_chunk_per_op():
    ops = [
        MergeOp(type="symmetric", primary=_chunk("c1"), secondary=_chunk("c2")),
        MergeOp(type="asymmetric", primary=_chunk("c3"), secondary=_chunk("c4")),
    ]
    plan = MergePlan(operations=ops)
    results = execute(plan, query="what happened?", llm=_mock_llm())
    assert len(results) == 2


def test_merged_chunk_preserves_source_ids():
    op = MergeOp(type="symmetric", primary=_chunk("c1"), secondary=_chunk("c2"))
    plan = MergePlan(operations=[op])
    results = execute(plan, query="q", llm=_mock_llm())
    assert set(results[0].source_chunk_ids) == {"c1", "c2"}


def test_merged_chunk_has_correct_type():
    op = MergeOp(type="asymmetric", primary=_chunk("c1"), secondary=_chunk("c2"))
    plan = MergePlan(operations=[op])
    results = execute(plan, query="q", llm=_mock_llm())
    assert results[0].merge_type == "asymmetric"


def test_llm_called_once_per_op():
    ops = [MergeOp(type="asymmetric", primary=_chunk(f"c{i}"), secondary=_chunk(f"c{i+1}")) for i in range(3)]
    plan = MergePlan(operations=ops)
    llm = _mock_llm()
    execute(plan, query="q", llm=llm)
    assert llm.complete.call_count == 3


def test_empty_plan_returns_empty_list():
    results = execute(MergePlan(operations=[]), query="q", llm=_mock_llm())
    assert results == []
