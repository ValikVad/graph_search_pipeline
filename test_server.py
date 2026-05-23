#!/usr/bin/env python3
"""
test_server.py — Test GraphSearch MCP tools without running an MCP transport.

Calls the tool functions directly (they are plain Python functions under the hood)
so you can verify everything works before connecting a real MCP client.

Usage:
    python test_server.py                              # smoke-test all tools
    python test_server.py --repo karpathy_nanochat     # one specific repo
    python test_server.py --query "how are embeddings computed" --repo karpathy_nanochat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import the tool functions directly from the server module.
# This also triggers the CLI arg parsing inside mcp_server, so we pass
# harmless defaults via sys.argv before the import.
sys.argv = ["mcp_server.py", "--ast-cache", "ast_cache"]

from mcp_server import (
    list_repos,
    repo_stats,
    search_code,
    search_code_explain,
    _available_repos,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _print_json(raw: str) -> None:
    try:
        parsed = json.loads(raw)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(raw)


def _check(label: str, raw: str) -> bool:
    """Print result and return True if no 'error' key."""
    try:
        data = json.loads(raw)
        ok = "error" not in data
    except json.JSONDecodeError:
        ok = False
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {status}  {label}")
    if not ok:
        _print_json(raw)
    return ok


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

def test_list_repos() -> list[str]:
    _section("Tool: list_repos()")
    result = list_repos()
    _print_json(result)
    data = json.loads(result)
    repos = data.get("available_repos", [])
    print(f"\n  → {len(repos)} repo(s) available")
    return repos


def test_repo_stats(repo: str) -> None:
    _section(f"Tool: repo_stats('{repo}')")
    result = repo_stats(repo)
    _print_json(result)


def test_search_code(repo: str, query: str, top_k: int = 5) -> None:
    _section(f"Tool: search_code(repo='{repo}', top_k={top_k})")
    print(f"  Query: \"{query}\"\n")

    t0 = time.time()
    result = search_code(query=query, repo=repo, top_k=top_k)
    elapsed = time.time() - t0

    data = json.loads(result)
    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return

    print(f"  Retrieved {data['results_count']} snippet(s) in {elapsed:.2f}s\n")
    for r in data["results"]:
        print(f"  [{r['rank']}] score={r['score']:.4f}  "
              f"{r['file']}:{r['lines']}  {r['qualified_name']}")
    print()

    # Show first result's code
    if data["results"]:
        top = data["results"][0]
        print("  ── Top result code ──────────────────────────────────────")
        for line in top["code"].splitlines()[:20]:
            print(f"    {line}")
        if len(top["code"].splitlines()) > 20:
            print("    ... (truncated for display)")
        print()


def test_search_explain(repo: str, query: str) -> None:
    _section(f"Tool: search_code_explain(repo='{repo}')")
    print(f"  Query: \"{query}\"\n")

    result = search_code_explain(query=query, repo=repo)
    data = json.loads(result)

    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return

    s1 = data["stage1_dual_encoder"]
    s2 = data["stage2_ast_expansion"]
    s3w = data["stage3_with_ast"]
    s3n = data["stage3_without_ast"]

    print(f"  Stage 1 — Dual Encoder (top-{s1['top_n']})")
    for c in s1["candidates"][:5]:
        print(f"    [{c['rank']}] {c['score']:.4f}  {c['file']}:{c['lines']}  {c['name']}")

    print(f"\n  Stage 2 — AST Expansion (hop_depth={s2['hop_depth']})")
    print(f"    New snippets added : {s2['new_snippets_added']}")
    print(f"    Total candidates   : {s2['total_candidates']}")
    if s2["added"]:
        print("    Sample added:")
        for a in s2["added"][:3]:
            print(f"      + {a['file']}:{a['lines']}  {a['name']}")

    print(f"\n  Stage 3 — Cross-Encoder WITH AST (top-{s3w['top_k']})")
    for r in s3w["results"]:
        rescue_tag = " ← AST rescue" if r.get("ast_rescue") else ""
        print(f"    [{r['rank']}] {r['score']:.4f}  "
              f"{r['file']}:{r['lines']}  {r['name']}{rescue_tag}")

    print(f"\n  Stage 3 — Cross-Encoder WITHOUT AST (baseline, top-{s3n['top_k']})")
    for r in s3n["results"]:
        print(f"    [{r['rank']}] {r['score']:.4f}  "
              f"{r['file']}:{r['lines']}  {r['name']}")

    # Delta summary
    with_ids    = {r["name"] for r in s3w["results"]}
    without_ids = {r["name"] for r in s3n["results"]}
    gained = with_ids - without_ids
    lost   = without_ids - with_ids
    if gained:
        print(f"\n  AST gained  : {gained}")
    if lost:
        print(f"  AST removed : {lost}")


def test_error_handling() -> None:
    _section("Error handling")
    bad_repo = search_code(query="anything", repo="nonexistent_repo_xyz")
    _check("nonexistent repo returns error JSON", bad_repo)


# ---------------------------------------------------------------------------
# Per-repo question sets (sampled from benchmark)
# ---------------------------------------------------------------------------
SAMPLE_QUERIES: dict[str, list[str]] = {
    "karpathy_nanochat": [
        "how are rotary position embeddings applied to queries and keys",
        "how does the sampling step turn logits into a next token",
    ],
    "microsoft_autogen": [
        "how does an agent decide to terminate a conversation",
        "how are tool call results passed back to the model",
    ],
    "vllm-project_vllm": [
        "how does the engine decide whether to run in-process or as a separate background process",
        "how are per-request token IDs assembled into GPU tensors before the forward pass",
    ],
    "infiniflow_ragflow": [
        "how are document chunks stored and retrieved from the vector index",
        "how does the system handle a failed embedding API call",
    ],
    "browser-use_browser-use": [
        "how does the agent decide which DOM element to interact with next",
        "how is a screenshot converted into a description the model can read",
    ],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Test GraphSearch MCP tools locally")
    ap.add_argument("--repo",  help="Repo to test (default: first available)")
    ap.add_argument("--query", help="Custom query to test with search_code")
    ap.add_argument("--explain", action="store_true",
                    help="Run search_code_explain instead of search_code")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    print("\nGraphSearch MCP — local tool test")
    print("──────────────────────────────────")

    # 1. list_repos
    repos = test_list_repos()
    if not repos:
        print("\n  No repos available. Run: python parse_ast.py")
        sys.exit(1)

    repo = args.repo or repos[0]
    if repo not in repos:
        print(f"\n  Repo '{repo}' not available. Choose from: {repos}")
        sys.exit(1)

    # 2. repo_stats
    test_repo_stats(repo)

    # 3. search_code  / search_code_explain
    queries = (
        [args.query] if args.query
        else SAMPLE_QUERIES.get(repo, ["how does the main entry point work"])
    )

    for q in queries:
        if args.explain:
            test_search_explain(repo, q)
        else:
            test_search_code(repo, q, top_k=args.top_k)

    # 4. Error handling (always runs)
    test_error_handling()

    print("\n── All tests done ──────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
