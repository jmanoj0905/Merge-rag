from mergerag.core.models import Chunk, MergeOp
from mergerag.core.utils import cosine_similarity


def assign_to_anchors(
    weak: list[Chunk],
    strong: list[Chunk],
    same_doc_only: bool = True,
    min_similarity: float = 0.3,
) -> list[MergeOp]:
    """Assign each weak chunk to its nearest same-doc strong anchor.

    Skips pairs where best cosine similarity is below min_similarity,
    preventing cross-document noise merges on a shared index.
    """
    if not weak or not strong:
        return []

    ops: list[MergeOp] = []
    for w in weak:
        candidates = [s for s in strong if s.doc_id == w.doc_id] if same_doc_only else list(strong)
        if not candidates:
            continue
        anchor = max(candidates, key=lambda s: cosine_similarity(w.embedding, s.embedding))
        if cosine_similarity(w.embedding, anchor.embedding) < min_similarity:
            continue
        ops.append(MergeOp(type="asymmetric", primary=anchor, secondary=w))

    return ops
