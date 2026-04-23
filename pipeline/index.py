from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx


class SnippetIndex:
    """Loads and holds all snippets + merged call graph for a set of repos."""

    def __init__(self, ast_cache_dir: Path) -> None:
        self._cache_dir = ast_cache_dir
        self.snippets: list[dict] = []
        self._id_to_snippet: dict[str, dict] = {}
        self._graph: nx.DiGraph = nx.DiGraph()

    def load(self, repos: list[str]) -> None:
        """Load snippets and call graphs for the given repo names."""
        self.snippets = []
        self._id_to_snippet = {}
        self._graph = nx.DiGraph()

        for repo in repos:
            repo_dir = self._cache_dir / repo
            snippets_path = repo_dir / "snippets.json"
            graph_path = repo_dir / "call_graph.pkl"

            if not snippets_path.exists():
                raise FileNotFoundError(
                    f"No snippets found for repo '{repo}'. "
                    f"Run parse_ast.py first."
                )

            with open(snippets_path, encoding="utf-8") as f:
                repo_snippets = json.load(f)
            self.snippets.extend(repo_snippets)
            for s in repo_snippets:
                self._id_to_snippet[s["id"]] = s

            if graph_path.exists():
                with open(graph_path, "rb") as f:
                    g: nx.DiGraph = pickle.load(f)
                self._graph = nx.compose(self._graph, g)

        print(
            f"[SnippetIndex] loaded {len(self.snippets)} snippets "
            f"from {len(repos)} repo(s), "
            f"{self._graph.number_of_nodes()} graph nodes"
        )

    def get_snippet(self, snippet_id: str) -> Optional[dict]:
        return self._id_to_snippet.get(snippet_id)

    def get_neighbors(self, snippet_id: str, depth: int) -> list[str]:
        """
        Return snippet IDs reachable within `depth` hops in either direction
        (callers + callees).  The source node is excluded from the result.
        """
        if depth == 0 or snippet_id not in self._graph:
            return []

        visited: set[str] = set()
        frontier = {snippet_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                next_frontier.update(self._graph.predecessors(nid))
                next_frontier.update(self._graph.successors(nid))
            next_frontier -= visited | {snippet_id}
            visited.update(next_frontier)
            frontier = next_frontier

        return list(visited)

    def available_repos(self) -> list[str]:
        return sorted(d.name for d in self._cache_dir.iterdir() if d.is_dir())
