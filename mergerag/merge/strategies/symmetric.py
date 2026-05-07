from mergerag.core.models import Chunk, MergeOp
from mergerag.core.utils import cosine_similarity


def pair_weak_chunks(weak: list[Chunk]) -> list[MergeOp]:
    """Greedily pair weak chunks by highest mutual cosine similarity."""
    if len(weak) < 2:
        return []

    ops: list[MergeOp] = []
    used: set[int] = set()

    for i in range(len(weak)):
        if i in used:
            continue
        best_j: int | None = None
        best_sim = -2.0
        for j in range(i + 1, len(weak)):
            if j in used:
                continue
            sim = cosine_similarity(weak[i].embedding, weak[j].embedding)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j is not None:
            ops.append(MergeOp(type="symmetric", primary=weak[i], secondary=weak[best_j]))
            used.add(i)
            used.add(best_j)

    return ops
