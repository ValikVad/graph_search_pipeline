"""
parse_ast.py — Parse Python repos into function-level snippets and call graphs.

Usage:
    python parse_ast.py                              # all repos
    python parse_ast.py --repo karpathy_nanochat     # single repo
    python parse_ast.py --repos repo1 repo2          # multiple repos
"""

import ast
import argparse
import json
import os
import pickle
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


REPOS_DIR = Path("python_repos")
CACHE_DIR = Path("ast_cache")

VIZ_FULL_THRESHOLD = 150   # draw full graph below this node count
VIZ_TOP_N = 80             # nodes to show for large graphs


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _get_call_names(node: ast.AST) -> list[str]:
    """Collect all names called within an AST node (shallow name resolution)."""
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            names.append(func.id)
        elif isinstance(func, ast.Attribute):
            # self.foo() → "foo";  obj.method() → "method"
            names.append(func.attr)
    return names


def _source_segment(source_lines: list[str], start: int, end: int) -> str:
    """Return source text for line range [start, end] (1-indexed, inclusive)."""
    segment = "".join(source_lines[start - 1 : end])
    return textwrap.dedent(segment)


# ---------------------------------------------------------------------------
# Per-file parsing
# ---------------------------------------------------------------------------

def parse_file(
    fpath: Path,
    repo_root: Path,
    repo_name: str,
) -> list[dict]:
    """Return a list of snippet dicts extracted from a single Python file."""
    try:
        source = fpath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    source_lines = source.splitlines(keepends=True)
    rel_path = str(fpath.relative_to(repo_root))
    snippets = []

    def visit(node: ast.AST, parent_class: Optional[str] = None):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                visit(child, parent_class=node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            qualified = f"{parent_class}.{name}" if parent_class else name
            node_type = "method" if parent_class else "function"
            start = node.lineno
            end = node.end_lineno or start

            # skip very short stubs (just a pass / docstring)
            if end - start < 1:
                return

            context = _source_segment(source_lines, start, end)
            calls = _get_call_names(node)

            snippet_id = f"{repo_name}::{rel_path}::{qualified}::L{start}-{end}"
            snippets.append({
                "id": snippet_id,
                "repo": repo_name,
                "file": rel_path,
                "name": name,
                "qualified_name": qualified,
                "node_type": node_type,
                "parent_class": parent_class,
                "start_line": start,
                "end_line": end,
                "context": context,
                "calls": calls,
            })
            # recurse for nested functions
            for child in node.body:
                visit(child, parent_class=parent_class)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, parent_class=parent_class)

    for node in ast.iter_child_nodes(tree):
        visit(node)

    return snippets


# ---------------------------------------------------------------------------
# Call graph construction
# ---------------------------------------------------------------------------

def build_call_graph(snippets: list[dict]) -> nx.DiGraph:
    """Build a directed call graph from snippet metadata."""
    # name → list of snippet IDs (multiple functions can share a name)
    name_to_ids: dict[str, list[str]] = {}
    for s in snippets:
        for key in (s["name"], s["qualified_name"]):
            name_to_ids.setdefault(key, []).append(s["id"])

    G = nx.DiGraph()
    for s in snippets:
        G.add_node(s["id"], label=s["qualified_name"], file=s["file"])

    for s in snippets:
        seen = set()
        for called_name in s["calls"]:
            if called_name in seen:
                continue
            seen.add(called_name)
            for callee_id in name_to_ids.get(called_name, []):
                if callee_id != s["id"]:
                    G.add_edge(s["id"], callee_id)

    return G


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _short_label(snippet_id: str) -> str:
    """Return a readable label: qualified_name (truncated)."""
    parts = snippet_id.split("::")
    return parts[2] if len(parts) >= 3 else snippet_id


def visualize(G: nx.DiGraph, out_path: Path) -> None:
    n = G.number_of_nodes()
    if n == 0:
        return

    if n >= VIZ_FULL_THRESHOLD:
        # keep only top-N nodes by total degree
        by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        keep = {nid for nid, _ in by_degree[:VIZ_TOP_N]}
        G = G.subgraph(keep).copy()
        n = G.number_of_nodes()

    in_deg = dict(G.in_degree())
    node_sizes = [300 + in_deg.get(v, 0) * 150 for v in G.nodes()]

    figsize = max(12, n // 4)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    pos = nx.spring_layout(G, seed=42, k=2.5 / max(n ** 0.5, 1))
    labels = {v: _short_label(v) for v in G.nodes()}

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#4C9BE8",
                           alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#888888", arrows=True,
                           arrowsize=12, width=0.8, ax=ax,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    ax.set_title(f"Call graph — {n} nodes shown", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-repo processing
# ---------------------------------------------------------------------------

def process_repo(repo_name: str) -> None:
    repo_root = REPOS_DIR / repo_name
    if not repo_root.is_dir():
        print(f"  [skip] {repo_name} — directory not found")
        return

    out_dir = CACHE_DIR / repo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    py_files = sorted(repo_root.rglob("*.py"))
    # skip .venv, __pycache__, site-packages, etc.
    py_files = [
        f for f in py_files
        if not any(part.startswith((".venv", "__pycache__", "site-packages", ".tox"))
                   for part in f.parts)
    ]

    snippets: list[dict] = []
    for fpath in py_files:
        snippets.extend(parse_file(fpath, repo_root, repo_name))

    print(f"  {repo_name}: {len(py_files)} files, {len(snippets)} snippets")

    # save snippets
    with open(out_dir / "snippets.json", "w", encoding="utf-8") as f:
        json.dump(snippets, f, indent=2)

    # build call graph
    G = build_call_graph(snippets)
    print(f"    call graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # save adjacency list (JSON-serializable)
    adj = {nid: list(G.successors(nid)) for nid in G.nodes()}
    with open(out_dir / "call_graph.json", "w", encoding="utf-8") as f:
        json.dump(adj, f, indent=2)

    # save networkx graph
    with open(out_dir / "call_graph.pkl", "wb") as f:
        pickle.dump(G, f)

    # visualization
    visualize(G, out_dir / "call_graph.png")
    print(f"    saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--repo", help="Single repo name (owner_repo)")
    group.add_argument("--repos", nargs="+", help="Multiple repo names")
    args = parser.parse_args()

    if args.repo:
        repos = [args.repo]
    elif args.repos:
        repos = args.repos
    else:
        repos = sorted(d.name for d in REPOS_DIR.iterdir() if d.is_dir()
                       and d.name != "repo_metadata.json" and not d.name.startswith("."))

    print(f"Processing {len(repos)} repo(s)...")
    for repo in repos:
        process_repo(repo)

    print("Done.")


if __name__ == "__main__":
    main()
