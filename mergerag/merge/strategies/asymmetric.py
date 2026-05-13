from mergerag.core.models import Chunk, MergeOp
from mergerag.core.utils import cosine_similarity


def assign_to_anchors(
    weak: list[Chunk],
    strong: list[Chunk],
    same_doc_only: bool = True,
    min_similarity: float = 0.3,
    max_ops: int = 3,
) -> list[MergeOp]:
    """Assign weak chunks to their nearest same-doc strong anchor.

    Skips pairs below min_similarity. Caps at max_ops total merge LLM calls
    to bound latency — takes the highest-similarity pairs first.
    """
    if not weak or not strong:
        return []

    candidates_list: list[tuple[float, Chunk, Chunk]] = []
    for w in weak:
        anchors = [s for s in strong if s.doc_id == w.doc_id] if same_doc_only else list(strong)
        if not anchors:
            continue
        anchor = max(anchors, key=lambda s: cosine_similarity(w.embedding, s.embedding))
        sim = cosine_similarity(w.embedding, anchor.embedding)
        if sim >= min_similarity:
            candidates_list.append((sim, anchor, w))

    candidates_list.sort(key=lambda x: x[0], reverse=True)

    return [
        MergeOp(type="asymmetric", primary=anchor, secondary=w)
        for _, anchor, w in candidates_list[:max_ops]
    ]
