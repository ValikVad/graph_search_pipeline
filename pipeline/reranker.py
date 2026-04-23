from __future__ import annotations

from typing import Optional

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Reranks candidates with a cross-encoder and returns the top-k."""

    def __init__(self, model_name: str, device: str) -> None:
        print(f"[CrossEncoderReranker] loading '{model_name}' on {device}")
        self._model = CrossEncoder(model_name, device=device)

    @staticmethod
    def _snippet_text(snippet: dict) -> str:
        header = f"# {snippet['file']} — {snippet['qualified_name']}"
        return f"{header}\n{snippet['context']}"

    def rerank(
        self,
        query: str,
        candidates: list[tuple[dict, Optional[float]]],
        top_k: int,
    ) -> list[tuple[dict, float]]:
        """Return top-k (snippet, score) sorted by cross-encoder score descending."""
        if not candidates:
            return []

        pairs = [(query, self._snippet_text(s)) for s, _ in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip([s for s, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
