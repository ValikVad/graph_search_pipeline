from __future__ import annotations

from typing import Optional

from .index import SnippetIndex


class ASTNeighborExpander:
    """
    Expands a candidate list by adding call-graph neighbours up to `hop_depth`
    hops away (bidirectional: callers + callees).

    Seeds keep their dual-encoder score; added neighbours get score=None.
    """

    def __init__(self, index: SnippetIndex, hop_depth: int) -> None:
        self._index = index
        self._hop_depth = hop_depth

    def expand(
        self, candidates: list[tuple[dict, float]]
    ) -> list[tuple[dict, Optional[float]]]:
        """Return a deduplicated list of (snippet, score_or_None)."""
        if self._hop_depth == 0:
            return candidates  # type: ignore[return-value]

        seen_ids: set[str] = {s["id"] for s, _ in candidates}
        expanded: list[tuple[dict, Optional[float]]] = list(candidates)

        for snippet, _ in candidates:
            for nid in self._index.get_neighbors(snippet["id"], self._hop_depth):
                if nid in seen_ids:
                    continue
                neighbor = self._index.get_snippet(nid)
                if neighbor is not None:
                    expanded.append((neighbor, None))
                    seen_ids.add(nid)

        return expanded
