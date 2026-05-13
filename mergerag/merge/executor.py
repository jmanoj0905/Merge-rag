import uuid
from pathlib import Path

from mergerag.core.models import MergePlan, MergedChunk
from mergerag.core.ports import LLMPort

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(merge_type: str) -> str:
    return (_PROMPTS_DIR / f"merge_{merge_type}.txt").read_text()


def _concatenate(primary_text: str, secondary_text: str) -> str:
    return primary_text.rstrip() + "\n\n" + secondary_text.lstrip()


def execute(plan: MergePlan, query: str, llm: LLMPort) -> list[MergedChunk]:
    """Merge chunks per plan. Symmetric uses concatenation; asymmetric uses LLM synthesis."""
    merged: list[MergedChunk] = []
    for op in plan.operations:
        if op.type == "symmetric":
            # Concatenation avoids hallucination from asking a small LLM to synthesize
            # two low-relevance weak chunks — empirically caused wrong answers in n=500 run.
            text = _concatenate(op.primary.text, op.secondary.text)
        else:
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
