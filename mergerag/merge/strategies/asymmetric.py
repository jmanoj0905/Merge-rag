from mergerag.core.models import Chunk, MergeOp
from mergerag.core.utils import cosine_similarity


def assign_to_anchors(weak: list[Chunk], strong: list[Chunk]) -> list[MergeOp]:
    """Assign each weak chunk to its nearest strong anchor by cosine similarity."""
    if not weak or not strong:
        return []

    ops: list[MergeOp] = []
    for w in weak:
        anchor = max(strong, key=lambda s: cosine_similarity(w.embedding, s.embedding))
        ops.append(MergeOp(type="asymmetric", primary=anchor, secondary=w))

    return ops
