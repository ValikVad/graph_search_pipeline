"""
run_eval.py — Parse all repos and evaluate against all question datasets.

Saves aggregate metrics to reports/{repo}.json and per-question details to
full_reports/{repo}.json.

Usage:
    python run_eval.py                   # all repos with questions
    python run_eval.py --repos karpathy_nanochat browser-use_browser-use
    python run_eval.py --skip-parse      # skip parse_ast step (ast_cache already built)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate import evaluate, is_match
from parse_ast import process_repo
from pipeline import PipelineConfig
from pipeline.dual_encoder import DualEncoder
from pipeline.expander import ASTNeighborExpander
from pipeline.index import SnippetIndex
from pipeline.reranker import CrossEncoderReranker

REPOS_DIR    = Path("python_repos")
QUESTIONS_DIR     = Path("questions")
QUESTIONS_HARD_DIR = Path("questions_hard")
AST_CACHE_DIR = Path("ast_cache")
REPORTS_DIR   = Path("reports")
FULL_REPORTS_DIR = Path("full_reports")


def repos_with_questions() -> list[str]:
    return sorted(
        p.stem for p in QUESTIONS_DIR.glob("*.json")
        if (REPOS_DIR / p.stem).is_dir()
    )


def run(
    repos: list[str],
    cfg: PipelineConfig,
    skip_parse: bool,
    verbose: bool,
) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    device = cfg.resolved_device()

    # shared cross-encoder (heavy to load — reuse across repos)
    reranker = CrossEncoderReranker(cfg.cross_encoder_model, device)

    for repo in repos:
        print(f"\n{'='*70}")
        print(f"REPO: {repo}")
        print(f"{'='*70}")

        # --- parse ---
        if not skip_parse:
            process_repo(repo)
        elif not (AST_CACHE_DIR / repo / "snippets.json").exists():
            print(f"  [skip] ast_cache not found and --skip-parse is set")
            continue

        # --- index + encode ---
        index = SnippetIndex(AST_CACHE_DIR)
        index.load([repo])

        dual_encoder = DualEncoder(cfg.dual_encoder_model, device, cfg.batch_size)
        dual_encoder.index(index.snippets)

        expander = ASTNeighborExpander(index, cfg.hop_depth)

        # --- evaluate question sets ---
        run_config = {
            "dual_encoder_model": cfg.dual_encoder_model,
            "cross_encoder_model": cfg.cross_encoder_model,
            "top_n": cfg.top_n,
            "hop_depth": cfg.hop_depth,
            "top_k": cfg.top_k,
        }
        evaluated_at = datetime.now(timezone.utc).isoformat()
        report: dict = {
            "repo": repo,
            "n_snippets": len(index.snippets),
            "config": run_config,
            "evaluated_at": evaluated_at,
            "questions": {},
        }
        all_records: list[dict] = []

        for split, qdir in [("easy", QUESTIONS_DIR), ("hard", QUESTIONS_HARD_DIR)]:
            qfile = qdir / f"{repo}.json"
            if not qfile.exists():
                continue
            questions = json.loads(qfile.read_text())
            print(f"\n--- {split.upper()} ({len(questions)} questions) ---")
            metrics, records = evaluate(
                questions, index, dual_encoder, expander, reranker, cfg,
                verbose=verbose, split=split,
            )
            report["questions"][split] = {"n": len(questions), **metrics}
            all_records.extend(records)

        # --- save aggregate report ---
        out = REPORTS_DIR / f"{repo}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\n  Report saved → {out}")

        # --- save full per-question report ---
        FULL_REPORTS_DIR.mkdir(exist_ok=True)
        full_out = FULL_REPORTS_DIR / f"{repo}.json"
        full_report = {
            "repo": repo,
            "n_snippets": len(index.snippets),
            "config": run_config,
            "evaluated_at": evaluated_at,
            "records": all_records,
        }
        full_out.write_text(json.dumps(full_report, indent=2))
        print(f"  Full report saved → {full_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse + evaluate all repos")
    parser.add_argument("--repos", nargs="+", default=None,
                        help="Repos to process (default: all with question files)")
    parser.add_argument("--skip-parse", action="store_true",
                        help="Skip parse_ast step (use existing ast_cache)")
    parser.add_argument("--top-n",     type=int, default=20)
    parser.add_argument("--hop-depth", type=int, default=1)
    parser.add_argument("--top-k",     type=int, default=10)
    parser.add_argument("--dual-model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-model",
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repos = args.repos or repos_with_questions()
    if not repos:
        print("No repos found with question files.", file=sys.stderr)
        sys.exit(1)

    print(f"Repos to evaluate: {repos}")

    cfg = PipelineConfig(
        dual_encoder_model=args.dual_model,
        cross_encoder_model=args.cross_model,
        top_n=args.top_n,
        hop_depth=args.hop_depth,
        top_k=args.top_k,
    )

    run(repos, cfg, skip_parse=args.skip_parse, verbose=args.verbose)
    print("\nAll done.")


if __name__ == "__main__":
    main()
