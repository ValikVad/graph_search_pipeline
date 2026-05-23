"""
pipeline/bm25_retriever.py
--------------------------
BM25 retriever as a drop-in replacement for DualEncoder.
Uses rank_bm25 over tokenized snippet text.

Install: pip install rank-bm25
"""
from __future__ import annotations

import re
from typing import Optional

from ...graphsearch.pipeline.index import SnippetIndex


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for code."""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", text.lower())


def _snippet_text(snippet: dict) -> str:
    header = f"{snippet['file']} {snippet['qualified_name']}"
    return f"{header}\n{snippet['context']}"


class BM25Retriever:
    """
    BM25 retriever over function-level snippets.
    API matches DualEncoder so it can be swapped in evaluate.py.
    """

    def __init__(self) -> None:
        self._snippets: list[dict] = []
        self._bm25 = None

    def index(self, snippets: list[dict]) -> None:
        from rank_bm25 import BM25Okapi          # pip install rank-bm25

        self._snippets = snippets
        corpus = [_tokenize(_snippet_text(s)) for s in snippets]
        print(f"[BM25Retriever] indexing {len(corpus)} snippets…")
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_n: int) -> list[tuple[dict, float]]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before search().")

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        k = min(top_n, len(scores))
        import numpy as np
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(self._snippets[i], float(scores[i])) for i in top_idx]
