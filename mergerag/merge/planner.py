from typing import Literal
from mergerag.core.models import Chunk, MergePlan
from mergerag.merge.strategies.symmetric import pair_weak_chunks
from mergerag.merge.strategies.asymmetric import assign_to_anchors


def plan(
    chunks: list[Chunk],
    strategy: Literal["symmetric", "asymmetric"],
    strong_k: int,
) -> MergePlan:
    """Split chunks by rank and build a MergePlan. No I/O."""
    strong = chunks[:strong_k]
    weak = chunks[strong_k:]

    if not weak:
        return MergePlan(operations=[])

    if strategy == "symmetric":
        ops = pair_weak_chunks(weak)
    else:
        ops = assign_to_anchors(weak, strong)

    return MergePlan(operations=ops)
