#!/usr/bin/env python3
"""
GraphSearch MCP Server
======================
Exposes the GraphSearch three-stage code search pipeline as MCP tools.
Any MCP-compatible client (Claude Desktop, Claude Code, Cursor, etc.)
can use these tools to search code in real Python repositories by
developer intent.

Pipeline stages
---------------
  Stage 1 — Dual Encoder      : bi-encoder top-N by cosine similarity
  Stage 2 — AST Expansion     : add call-graph neighbours (callers + callees)
  Stage 3 — Cross-Encoder     : rerank the expanded set, return top-K

Quick start
-----------
  # 1. Parse repos (if not done yet)
  python parse_ast.py

  # 2. Run the server (stdio — for Claude Desktop / Claude Code)
  python mcp_server.py

  # 3. Optional: preload heavy repos at startup to avoid first-query lag
  python mcp_server.py --preload vllm-project_vllm infiniflow_ragflow

  # 4. SSE mode (for web clients)
  python mcp_server.py --transport sse --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Parse CLI args before anything else so --help works without heavy imports
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="GraphSearch MCP Server",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument(
    "--ast-cache", default="ast_cache",
    help="Path to ast_cache directory produced by parse_ast.py",
)
_parser.add_argument("--top-n",     type=int, default=20,
                     help="Dual-encoder candidate count per query")
_parser.add_argument("--hop-depth", type=int, default=1,
                     help="Call-graph expansion depth (0 = disabled)")
_parser.add_argument("--top-k",     type=int, default=5,
                     help="Final results returned per query")
_parser.add_argument(
    "--transport", default="stdio", choices=["stdio", "sse"],
    help="MCP transport protocol",
)
_parser.add_argument("--port", type=int, default=8000,
                     help="Port for SSE transport")
_parser.add_argument(
    "--preload", nargs="*", metavar="REPO",
    help="Repos to index at startup (avoids first-query delay)",
)
_args, _ = _parser.parse_known_args()

AST_CACHE_DIR  = Path(_args.ast_cache)
DEFAULT_TOP_N  = _args.top_n
DEFAULT_HOP    = _args.hop_depth
DEFAULT_TOP_K  = _args.top_k

# ---------------------------------------------------------------------------
# Add repo root to sys.path so `pipeline` package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP           # pip install mcp
from pipeline import CodeSearchPipeline, PipelineConfig  # local package
from pipeline.index import SnippetIndex

# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="GraphSearch",
    instructions="""
You have access to a repository-level code search engine (GraphSearch).
It uses a three-stage pipeline: dense bi-encoder retrieval, AST call-graph
expansion, and cross-encoder reranking.

Workflow:
  1. Call list_repos() to see what is available.
  2. Optionally call repo_stats(repo) to gauge corpus size.
  3. Call search_code(query, repo) with your intent phrased as a question
     — never name the specific function you are looking for.
  4. Use search_code_explain(query, repo) when you want to understand
     WHY a snippet was or was not retrieved.

Good query examples:
  "how does the system fall back when the primary LLM is unavailable"
  "where are GPU tensors assembled before the model forward pass"
  "how are incoming API requests authenticated"
""",
)

# ---------------------------------------------------------------------------
# Pipeline cache — one instance per repo, built lazily
# ---------------------------------------------------------------------------
_cache: dict[str, CodeSearchPipeline] = {}


def _available_repos() -> list[str]:
    """Return repo names that have a parsed snippets.json in ast_cache."""
    if not AST_CACHE_DIR.exists():
        return []
    return sorted(
        d.name
        for d in AST_CACHE_DIR.iterdir()
        if d.is_dir() and (d / "snippets.json").exists()
    )


def _load_pipeline(repo: str) -> CodeSearchPipeline:
    """Return a ready pipeline for *repo*, indexing it on first call."""
    if repo not in _cache:
        snippets_path = AST_CACHE_DIR / repo / "snippets.json"
        if not snippets_path.exists():
            available = _available_repos()
            raise FileNotFoundError(
                f"Repository '{repo}' not found in {AST_CACHE_DIR}.\n"
                f"Run:  python parse_ast.py --repo {repo}\n"
                f"Available: {available or '(none — run parse_ast.py first)'}"
            )
        cfg = PipelineConfig(
            top_n=DEFAULT_TOP_N,
            hop_depth=DEFAULT_HOP,
            top_k=DEFAULT_TOP_K,
        )
        pipeline = CodeSearchPipeline(cfg, AST_CACHE_DIR)
        pipeline.load([repo])
        _cache[repo] = pipeline

    return _cache[repo]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
_MAX_CODE_CHARS = 1500  # truncate long snippets in tool output


def _fmt_snippet(snippet: dict, score: float, rank: int) -> dict:
    code = snippet["context"]
    truncated = False
    if len(code) > _MAX_CODE_CHARS:
        code = code[:_MAX_CODE_CHARS] + "\n... [truncated]"
        truncated = True
    return {
        "rank":           rank,
        "score":          round(score, 4),
        "file":           snippet["file"],
        "qualified_name": snippet["qualified_name"],
        "lines":          f"{snippet['start_line']}-{snippet['end_line']}",
        "node_type":      snippet.get("node_type", "function"),
        "code":           code,
        "truncated":      truncated,
    }


# ---------------------------------------------------------------------------
# Tool: list_repos
# ---------------------------------------------------------------------------
@mcp.tool()
def list_repos() -> str:
    """
    List all repositories available for code search.

    Returns a JSON object with the list of repo names that can be passed
    to search_code or repo_stats.
    """
    repos = _available_repos()
    if not repos:
        return json.dumps({
            "error": f"No repositories found in '{AST_CACHE_DIR}'.",
            "hint":  "Run: python parse_ast.py",
        }, indent=2)

    return json.dumps(
        {"available_repos": repos, "count": len(repos)},
        indent=2,
    )


# ---------------------------------------------------------------------------
# Tool: repo_stats
# ---------------------------------------------------------------------------
@mcp.tool()
def repo_stats(repo: str) -> str:
    """
    Return statistics about a repository: total snippets, file count,
    and the ten files with the most functions.

    Use this before search_code to understand corpus density.

    Args:
        repo: Repository name, e.g. 'vllm-project_vllm'.
              Use list_repos() to see valid names.
    """
    snippets_path = AST_CACHE_DIR / repo / "snippets.json"
    if not snippets_path.exists():
        return json.dumps({
            "error":     f"Repository '{repo}' not found.",
            "available": _available_repos(),
        }, indent=2)

    with open(snippets_path, encoding="utf-8") as fh:
        snippets: list[dict] = json.load(fh)

    file_counts: dict[str, int] = {}
    for s in snippets:
        file_counts[s["file"]] = file_counts.get(s["file"], 0) + 1

    top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return json.dumps({
        "repo":            repo,
        "total_snippets":  len(snippets),
        "total_files":     len(file_counts),
        "top_files": [{"file": f, "snippets": n} for f, n in top_files],
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: search_code
# ---------------------------------------------------------------------------
@mcp.tool()
def search_code(query: str, repo: str, top_k: int = 5) -> str:
    """
    Search for relevant code snippets in a repository by developer intent.

    Runs the full three-stage pipeline:
      1. Bi-encoder retrieves top-N candidates by embedding similarity.
      2. AST call-graph expansion adds callers/callees of those candidates.
      3. Cross-encoder reranks the expanded set and returns top-K.

    Args:
        query:  What you want to find, expressed as developer intent.
                Phrase it as a question or goal — do NOT name the function.
                Good:  "how does the scheduler decide which request runs next"
                Bad:   "find the _schedule function"
        repo:   Repository name from list_repos(), e.g. 'microsoft_autogen'.
        top_k:  Number of results to return (1–20, default 5).

    Returns:
        JSON with ranked snippets, each containing file path, line range,
        qualified function name, relevance score, and source code.
    """
    top_k = max(1, min(int(top_k), 20))

    try:
        pipeline = _load_pipeline(repo)
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Pipeline error: {exc}"}, indent=2)

    # Temporarily override top_k so the caller can vary it per-query
    original_k = pipeline.config.top_k
    pipeline.config.top_k = top_k
    try:
        results = pipeline.search(query)
    except Exception as exc:
        return json.dumps({"error": f"Search error: {exc}"}, indent=2)
    finally:
        pipeline.config.top_k = original_k

    snippets = [_fmt_snippet(s, score, rank=i + 1)
                for i, (s, score) in enumerate(results)]

    return json.dumps({
        "query":   query,
        "repo":    repo,
        "top_k":   top_k,
        "results": snippets,
    }, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: search_code_explain
# ---------------------------------------------------------------------------
@mcp.tool()
def search_code_explain(query: str, repo: str) -> str:
    """
    Search for code AND show what each pipeline stage contributed.

    Returns an annotated trace showing:
      - Stage 1: which snippets the bi-encoder ranked highest
      - Stage 2: which new snippets AST expansion added (and from where)
      - Stage 3: the final cross-encoder ranked results

    Useful for debugging missed retrievals, understanding call-graph effects,
    or generating paper-quality pipeline traces.

    Args:
        query: Developer intent (same format as search_code).
        repo:  Repository name from list_repos().
    """
    try:
        pipeline = _load_pipeline(repo)
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    index    = pipeline._index
    encoder  = pipeline._dual_encoder
    expander = pipeline._expander
    reranker = pipeline._reranker
    cfg      = pipeline.config

    # ---- Stage 1 ----
    stage1 = encoder.search(query, cfg.top_n)
    stage1_ids = {s["id"] for s, _ in stage1}

    stage1_preview = [
        {
            "rank":  i + 1,
            "score": round(sc, 4),
            "file":  s["file"],
            "name":  s["qualified_name"],
            "lines": f"{s['start_line']}-{s['end_line']}",
        }
        for i, (s, sc) in enumerate(stage1[:10])
    ]

    # ---- Stage 2 ----
    expanded      = expander.expand(stage1)
    new_ids       = {s["id"] for s, _ in expanded} - stage1_ids
    added_snippets = [
        {
            "file":  s["file"],
            "name":  s["qualified_name"],
            "lines": f"{s['start_line']}-{s['end_line']}",
            "note":  "added by call-graph expansion",
        }
        for s, _ in expanded
        if s["id"] in new_ids
    ]

    # ---- Stage 3 (with AST) ----
    results_with = reranker.rerank(query, expanded, cfg.top_k)

    # ---- Stage 3 (without AST, for comparison) ----
    results_without = reranker.rerank(query, stage1, cfg.top_k)

    def _brief_results(ranked):
        return [
            {
                "rank":  i + 1,
                "score": round(sc, 4),
                "file":  s["file"],
                "name":  s["qualified_name"],
                "lines": f"{s['start_line']}-{s['end_line']}",
                "ast_rescue": s["id"] in new_ids,
            }
            for i, (s, sc) in enumerate(ranked)
        ]

    return json.dumps({
        "query": query,
        "repo":  repo,
        "stage1_dual_encoder": {
            "top_n":      cfg.top_n,
            "candidates": stage1_preview,
        },
        "stage2_ast_expansion": {
            "hop_depth":        cfg.hop_depth,
            "new_snippets_added": len(new_ids),
            "total_candidates": len(expanded),
            "added":            added_snippets[:10],
        },
        "stage3_with_ast": {
            "top_k":   cfg.top_k,
            "results": _brief_results(results_with),
        },
        "stage3_without_ast": {
            "top_k":   cfg.top_k,
            "results": _brief_results(results_without),
            "note":    "baseline — same reranker but no call-graph expansion",
        },
    }, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Preload repos at startup if requested
# ---------------------------------------------------------------------------
if _args.preload:
    for _repo in _args.preload:
        print(f"[graphsearch] preloading '{_repo}' …", file=sys.stderr)
        try:
            _load_pipeline(_repo)
            print(f"[graphsearch] '{_repo}' ready.", file=sys.stderr)
        except FileNotFoundError as _e:
            print(f"[graphsearch] WARNING: {_e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if _args.transport == "sse":
        mcp.run(transport="sse", port=_args.port)
    else:
        mcp.run(transport="stdio")
