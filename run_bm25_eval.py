"""
run_bm25_eval.py
----------------
Evaluate BM25 retriever (no neural models) as a baseline.
Runs on all repos and both question splits, saves summary to
bm25_reports/summary.json.

Usage:
    pip install rank-bm25
    python run_bm25_eval.py
    python run_bm25_eval.py --repos karpathy_nanochat microsoft_autogen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.index import SnippetIndex
from pipeline.bm25_retriever import BM25Retriever
from pipeline.expander import ASTNeighborExpander
from pipeline.reranker import CrossEncoderReranker
from pipeline.config import PipelineConfig
from evaluate import evaluate, is_match

QUESTIONS_DIR      = Path("questions")
QUESTIONS_HARD_DIR = Path("questions_hard")
AST_CACHE_DIR      = Path("ast_cache")
OUT_DIR            = Path("bm25_reports")


def run_repo(repo: str, cfg: PipelineConfig) -> dict:
    index = SnippetIndex(AST_CACHE_DIR)
    index.load([repo])

    bm25     = BM25Retriever()
    bm25.index(index.snippets)

    expander = ASTNeighborExpander(index, cfg.hop_depth)
    reranker = CrossEncoderReranker(cfg.cross_encoder_model,
                                    cfg.resolved_device())

    result = {"repo": repo, "n_snippets": len(index.snippets), "splits": {}}

    for split_name, qdir in [("easy", QUESTIONS_DIR),
                              ("hard", QUESTIONS_HARD_DIR)]:
        qpath = qdir / f"{repo}.json"
        if not qpath.exists():
            continue
        questions = json.loads(qpath.read_text())

        # ---- Stage 1: BM25 top-N ----
        s1_found = 0
        s3_found = 0
        for q in questions:
            gt_file, gt_s, gt_e = q["file"], q["line_start"], q["line_end"]

            stage1 = bm25.search(q["question"], cfg.top_n)
            if any(is_match(s, gt_file, gt_s, gt_e) for s, _ in stage1):
                s1_found += 1

            results = reranker.rerank(q["question"], stage1, cfg.top_k)
            if any(is_match(s, gt_file, gt_s, gt_e) for s, _ in results):
                s3_found += 1

        n = len(questions)
        result["splits"][split_name] = {
            "n": n,
            "bm25_top_n":     round(s1_found / n, 4),
            "bm25_reranked":  round(s3_found / n, 4),
        }
        print(f"  {repo}:{split_name}  "
              f"BM25@{cfg.top_n}={s1_found}/{n}  "
              f"BM25+rerank@{cfg.top_k}={s3_found}/{n}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*",
                    help="Repos to evaluate (default: all in ast_cache)")
    ap.add_argument("--top-n",     type=int, default=20)
    ap.add_argument("--hop-depth", type=int, default=0,
                    help="BM25 baseline: no AST expansion (set 0)")
    ap.add_argument("--top-k",     type=int, default=5)
    args = ap.parse_args()

    cfg = PipelineConfig(top_n=args.top_n, hop_depth=args.hop_depth,
                         top_k=args.top_k)

    repos = args.repos or sorted(
        d.name for d in AST_CACHE_DIR.iterdir()
        if d.is_dir() and (d / "snippets.json").exists()
    )

    OUT_DIR.mkdir(exist_ok=True)
    all_results = []

    for repo in repos:
        print(f"\n[{repo}]")
        res = run_repo(repo, cfg)
        all_results.append(res)

    # Aggregate summary
    totals: dict[str, dict[str, int]] = {"easy": {"n": 0, "s1": 0, "s3": 0},
                                          "hard": {"n": 0, "s1": 0, "s3": 0}}
    for res in all_results:
        for split, data in res["splits"].items():
            n = data["n"]
            totals[split]["n"]  += n
            totals[split]["s1"] += round(data["bm25_top_n"] * n)
            totals[split]["s3"] += round(data["bm25_reranked"] * n)

    summary = {
        "config": {"top_n": cfg.top_n, "top_k": cfg.top_k,
                   "method": "BM25 (no AST expansion)"},
        "overall": {
            split: {
                "n":             t["n"],
                "bm25_hit":      round(t["s1"] / max(t["n"], 1), 4),
                "bm25_reranked": round(t["s3"] / max(t["n"], 1), 4),
            }
            for split, t in totals.items()
        },
        "per_repo": all_results,
    }

    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_path}")

    print("\n── Summary ──────────────────────────────────────────")
    for split, data in summary["overall"].items():
        print(f"  {split:5s}: BM25@{cfg.top_n}={data['bm25_hit']:.1%}"
              f"  BM25+rerank@{cfg.top_k}={data['bm25_reranked']:.1%}")


if __name__ == "__main__":
    main()
