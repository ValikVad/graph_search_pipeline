from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .dual_encoder import DualEncoder
from .expander import ASTNeighborExpander
from .index import SnippetIndex
from .reranker import CrossEncoderReranker
from .visualize import visualize_search


class CodeSearchPipeline:
    """
    Three-stage code search pipeline:
      1. DualEncoder         — cosine similarity, top-n candidates
      2. ASTNeighborExpander — d-hop call-graph expansion
      3. CrossEncoderReranker — rerank, return top-k

    Example
    -------
    cfg = PipelineConfig(top_n=20, hop_depth=1, top_k=5)
    pipeline = CodeSearchPipeline(cfg, Path("ast_cache"))
    pipeline.load(["karpathy_nanochat"])
    results = pipeline.search("how are rotary embeddings applied?")
    for snippet, score in results:
        print(score, snippet["id"])

    # optional: save a graph visualisation of the last search
    pipeline.visualize_last_search(Path("search_graph.png"))
    """

    def __init__(
        self,
        config: PipelineConfig,
        ast_cache_dir: Path = Path("ast_cache"),
    ) -> None:
        self.config = config
        device = config.resolved_device()

        self._index = SnippetIndex(ast_cache_dir)
        self._dual_encoder = DualEncoder(
            config.dual_encoder_model, device, config.batch_size
        )
        self._expander = ASTNeighborExpander(self._index, config.hop_depth)
        self._reranker = CrossEncoderReranker(config.cross_encoder_model, device)
        self._loaded = False
        self._last_stage_ids: Optional[tuple[set[str], set[str], set[str]]] = None

    def load(self, repos: Optional[list[str]] = None) -> None:
        """Load repos and build the embedding index. None → all available repos."""
        if repos is None:
            repos = self._index.available_repos()
        self._index.load(repos)
        self._dual_encoder.index(self._index.snippets)
        self._loaded = True

    def search(self, query: str) -> list[tuple[dict, float]]:
        """Run the full pipeline and return top-k (snippet, score) pairs."""
        if not self._loaded:
            raise RuntimeError("Call load() before search().")

        stage1 = self._dual_encoder.search(query, self.config.top_n)
        stage1_ids = {s["id"] for s, _ in stage1}

        expanded = self._expander.expand(stage1)
        expanded_ids = {s["id"] for s, _ in expanded} - stage1_ids

        results = self._reranker.rerank(query, expanded, self.config.top_k)
        topk_ids = {s["id"] for s, _ in results}

        self._last_stage_ids = (stage1_ids, expanded_ids, topk_ids)
        return results

    def visualize_last_search(self, out_path: Path) -> None:
        """Save a PNG of the call graph with nodes coloured by pipeline stage."""
        if self._last_stage_ids is None:
            raise RuntimeError("Call search() before visualize_last_search().")
        stage1_ids, expanded_ids, topk_ids = self._last_stage_ids
        visualize_search(
            self._index._graph,
            stage1_ids,
            expanded_ids,
            topk_ids,
            out_path,
        )
