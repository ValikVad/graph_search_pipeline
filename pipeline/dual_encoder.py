from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class DualEncoder:
    """
    Encodes a snippet corpus with a bi-encoder and retrieves the top-n most
    similar snippets for a query via cosine similarity (L2-normalised dot product).

    query_prefix is prepended to every query string before encoding.
    Corpus documents are never prefixed (asymmetric encoding).
    - MiniLM-style models : query_prefix=""  (default)
    - BAAI/bge-m3         : query_prefix="query: "
    - bge-large-en-v1.5   : query_prefix="Represent this sentence for searching relevant passages: "
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        query_prefix: str = "",
    ) -> None:
        print(f"[DualEncoder] loading '{model_name}' on {device}")
        if query_prefix:
            print(f"[DualEncoder] query_prefix='{query_prefix}'")
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._embeddings: Optional[np.ndarray] = None  # (N, D)
        self._snippets: list[dict] = []

    @staticmethod
    def _snippet_text(snippet: dict) -> str:
        header = f"# {snippet['file']} — {snippet['qualified_name']}"
        return f"{header}\n{snippet['context']}"

    def index(self, snippets: list[dict]) -> None:
        """Encode all snippets and cache embeddings for future searches."""
        self._snippets = snippets
        texts = [self._snippet_text(s) for s in snippets]
        print(f"[DualEncoder] encoding {len(texts)} snippets…")
        self._embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def search(self, query: str, top_n: int) -> list[tuple[dict, float]]:
        """Return top-n (snippet, cosine_score) pairs for the query."""
        if self._embeddings is None:
            raise RuntimeError("Call index() before search().")

        q_emb = self._model.encode(
            [self._query_prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores: np.ndarray = self._embeddings @ q_emb
        k = min(top_n, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(self._snippets[i], float(scores[i])) for i in top_idx]
