"""
evaluate.py — Evaluate the search pipeline against a questions dataset.

A retrieved snippet is a match when it covers the same file and its line range
overlaps with the ground-truth range.

Metrics reported per stage:
  Stage 1  : dual-encoder top-n (before expansion)
  Stage 2  : after AST neighbour expansion
  Stage 3  : cross-encoder top-k (final)

For each stage: Hit@1/3/5/10, MRR, mean rank of first hit.

Usage:
    python evaluate.py --questions questions/karpathy_nanochat.json \
                       --repo karpathy_nanochat
    python evaluate.py --questions questions/karpathy_nanochat.json \
                       --repo karpathy_nanochat \
                       --top-n 20 --hop-depth 1 --top-k 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline import CodeSearchPipeline, PipelineConfig
from pipeline.dual_encoder import DualEncoder
from pipeline.expander import ASTNeighborExpander
from pipeline.index import SnippetIndex
from pipeline.reranker import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def is_match(snippet: dict, gt_file: str, gt_start: int, gt_end: int) -> bool:
    """True if snippet covers the same file and lines overlap with GT range."""
    if snippet["file"] != gt_file:
        return False
    return snippet["start_line"] <= gt_end and snippet["end_line"] >= gt_start


# ---------------------------------------------------------------------------
# Per-stage metrics
# ---------------------------------------------------------------------------

@dataclass
class StageMetrics:
    name: str
    hits_at: dict[int, int] = field(default_factory=lambda: {1: 0, 3: 0, 5: 0, 10: 0})
    reciprocal_ranks: list[float] = field(default_factory=list)
    ranks: list[int] = field(default_factory=list)  # only when found
    n: int = 0

    def update(self, ranked_snippets: list[dict], gt_file: str,
               gt_start: int, gt_end: int) -> None:
        self.n += 1
        for rank, snippet in enumerate(ranked_snippets, 1):
            if is_match(snippet, gt_file, gt_start, gt_end):
                self.reciprocal_ranks.append(1.0 / rank)
                self.ranks.append(rank)
                for k in self.hits_at:
                    if rank <= k:
                        self.hits_at[k] += 1
                return
        self.reciprocal_ranks.append(0.0)

    def mrr(self) -> float:
        return sum(self.reciprocal_ranks) / self.n if self.n else 0.0

    def mean_rank(self) -> Optional[float]:
        return sum(self.ranks) / len(self.ranks) if self.ranks else None

    def to_dict(self) -> dict:
        mr = self.mean_rank()
        return {
            "n": self.n,
            "found": len(self.ranks),
            **{f"hit@{k}": round(self.hits_at[k] / self.n, 4) for k in sorted(self.hits_at)},
            "mrr": round(self.mrr(), 4),
            "mean_rank": round(mr, 2) if mr is not None else None,
        }

    def print(self) -> None:
        found = len(self.ranks)
        print(f"\n  [{self.name}]  n={self.n}  found={found}")
        ks = sorted(self.hits_at)
        hit_strs = "  ".join(f"Hit@{k}={self.hits_at[k]/self.n:.3f}" for k in ks)
        print(f"    {hit_strs}")
        mr = self.mean_rank()
        print(f"    MRR={self.mrr():.4f}  MeanRank={mr:.1f}" if mr else
              f"    MRR={self.mrr():.4f}  MeanRank=—")


# ---------------------------------------------------------------------------
# Evaluation loop (runs stages manually to capture intermediate results)
# ---------------------------------------------------------------------------

def _find_rank_score_id(
    ranked: list[tuple[dict, float]],
    gt_file: str,
    gt_start: int,
    gt_end: int,
) -> tuple[Optional[int], Optional[float], Optional[str]]:
    """Return (1-indexed rank, score, snippet id) of first GT match, or (None, None, None)."""
    for rank, (snippet, score) in enumerate(ranked, 1):
        if is_match(snippet, gt_file, gt_start, gt_end):
            return rank, score, snippet.get("id")
    return None, None, None


def evaluate(
    questions: list[dict],
    index: SnippetIndex,
    dual_encoder: DualEncoder,
    expander: ASTNeighborExpander,
    reranker: CrossEncoderReranker,
    config: PipelineConfig,
    verbose: bool = False,
    split: str = "",
) -> tuple[dict, list[dict]]:
    """Run evaluation and return (metrics dict, per-question records)."""
    m1 = StageMetrics("Stage 1 — dual encoder")
    m2 = StageMetrics("Stage 2 — AST expansion")
    m3 = StageMetrics("Stage 3 — reranking")
    m3_no_ast = StageMetrics("Ablation — reranking without AST expansion")

    records: list[dict] = []

    for i, q in enumerate(questions):
        gt_file  = q["file"]
        gt_start = q["line_start"]
        gt_end   = q["line_end"]
        query    = q["question"]

        if verbose:
            print(f"\n[{i+1}/{len(questions)}] {query}")

        stage1   = dual_encoder.search(query, config.top_n)
        m1.update([s for s, _ in stage1], gt_file, gt_start, gt_end)

        expanded = expander.expand(stage1)
        m2.update([s for s, _ in expanded], gt_file, gt_start, gt_end)

        results_no_ast = reranker.rerank(query, stage1, config.top_k)
        m3_no_ast.update([s for s, _ in results_no_ast], gt_file, gt_start, gt_end)

        results = reranker.rerank(query, expanded, config.top_k)
        m3.update([s for s, _ in results], gt_file, gt_start, gt_end)

        if verbose:
            hit = any(is_match(s, gt_file, gt_start, gt_end) for s, _ in results)
            print(f"  → {'HIT' if hit else 'MISS'}  ({gt_file} L{gt_start}-{gt_end})")

        s1_rank, s1_score, gt_id = _find_rank_score_id(stage1, gt_file, gt_start, gt_end)
        s2_found = any(is_match(s, gt_file, gt_start, gt_end) for s, _ in expanded)
        s3a_rank, s3a_score, gt_id_3a = _find_rank_score_id(results_no_ast, gt_file, gt_start, gt_end)
        s3_rank, s3_score, gt_id_3 = _find_rank_score_id(results, gt_file, gt_start, gt_end)

        records.append({
            "split": split,
            "idx": i,
            "question": query,
            "gt_file": gt_file,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "gt_id": gt_id_3 or gt_id_3a or gt_id,
            "stage1_rank": s1_rank,
            "stage1_score": round(s1_score, 4) if s1_score is not None else None,
            "stage2_found": s2_found,
            "stage3_no_ast_rank": s3a_rank,
            "stage3_no_ast_score": round(s3a_score, 4) if s3a_score is not None else None,
            "stage3_rank": s3_rank,
            "stage3_score": round(s3_score, 4) if s3_score is not None else None,
        })

    print("\n" + "=" * 60)
    print(f"Results  ({len(questions)} questions)")
    print("=" * 60)
    m1.print()
    m2.print()
    m3.print()
    m3_no_ast.print()
    print()

    return {
        "stage1_dual_encoder": m1.to_dict(),
        "stage2_ast_expanded": m2.to_dict(),
        "stage3_cross_encoder": m3.to_dict(),
        "ablation_rerank_without_ast": m3_no_ast.to_dict(),
    }, records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search pipeline")
    parser.add_argument("--questions", required=True,
                        help="Path to questions JSON file")
    parser.add_argument("--repo", required=True,
                        help="Repo name to search (must exist in ast_cache)")
    parser.add_argument("--top-n",     type=int, default=20)
    parser.add_argument("--hop-depth", type=int, default=1)
    parser.add_argument("--top-k",     type=int, default=5)
    parser.add_argument("--dual-model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-model",
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--ast-cache", default="ast_cache")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question hit/miss")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    print(f"Loaded {len(questions)} questions from {args.questions}")

    cfg = PipelineConfig(
        dual_encoder_model=args.dual_model,
        cross_encoder_model=args.cross_model,
        top_n=args.top_n,
        hop_depth=args.hop_depth,
        top_k=args.top_k,
    )
    device = cfg.resolved_device()

    index = SnippetIndex(Path(args.ast_cache))
    index.load([args.repo])

    dual_encoder = DualEncoder(cfg.dual_encoder_model, device, cfg.batch_size)
    dual_encoder.index(index.snippets)

    expander = ASTNeighborExpander(index, cfg.hop_depth)
    reranker = CrossEncoderReranker(cfg.cross_encoder_model, device)

    evaluate(questions, index, dual_encoder, expander, reranker, cfg,
             verbose=args.verbose)


if __name__ == "__main__":
    main()
