"""
search.py — CLI entrypoint for the code search pipeline.

Usage:
    python search.py "how are rotary embeddings applied?" --repos karpathy_nanochat
    python search.py "query" --top-n 20 --hop-depth 1 --top-k 5
"""

import argparse
from pathlib import Path

from pipeline import CodeSearchPipeline, PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Code search")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--repos", nargs="+", default=None,
                        help="Repo names to search (default: all in ast_cache)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Dual-encoder candidates (default: 20)")
    parser.add_argument("--hop-depth", type=int, default=1,
                        help="AST neighbour hops (default: 1)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Final results (default: 5)")
    parser.add_argument("--dual-model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-model",
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--ast-cache", default="ast_cache",
                        help="Path to ast_cache directory (default: ast_cache)")
    parser.add_argument("--viz", metavar="FILE",
                        help="Save a search graph visualisation to this PNG path")
    args = parser.parse_args()

    cfg = PipelineConfig(
        dual_encoder_model=args.dual_model,
        cross_encoder_model=args.cross_model,
        top_n=args.top_n,
        hop_depth=args.hop_depth,
        top_k=args.top_k,
    )
    pipeline = CodeSearchPipeline(cfg, ast_cache_dir=Path(args.ast_cache))
    pipeline.load(args.repos)

    print(f"\n=== Query: {args.query!r} ===\n")
    results = pipeline.search(args.query)

    if args.viz:
        pipeline.visualize_last_search(Path(args.viz))

    for rank, (snippet, score) in enumerate(results, 1):
        print(
            f"#{rank}  score={score:.4f}  [{snippet['node_type']}] "
            f"{snippet['qualified_name']}  "
            f"({snippet['file']} L{snippet['start_line']}-{snippet['end_line']})"
        )
        print("-" * 70)
        preview = snippet["context"][:300].rstrip()
        print(preview)
        if len(snippet["context"]) > 300:
            print("  …")
        print()


if __name__ == "__main__":
    main()
