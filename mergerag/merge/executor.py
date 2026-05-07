import uuid
from pathlib import Path

from mergerag.core.models import MergePlan, MergedChunk
from mergerag.core.ports import LLMPort

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(merge_type: str) -> str:
    return (_PROMPTS_DIR / f"merge_{merge_type}.txt").read_text()


def execute(plan: MergePlan, query: str, llm: LLMPort) -> list[MergedChunk]:
    """Call LLM once per MergeOp. Returns MergedChunks with source provenance."""
    merged: list[MergedChunk] = []
    for op in plan.operations:
        template = _load_prompt(op.type)
        prompt = template.format(
            query=query,
            chunk_a=op.primary.text,
            chunk_b=op.secondary.text,
        )
        text = llm.complete(prompt, max_tokens=512)
        merged.append(MergedChunk(
            id=str(uuid.uuid4()),
            text=text,
            score=0.0,
            source_chunk_ids=[op.primary.id, op.secondary.id],
            merge_type=op.type,
        ))
    return merged
