from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate import StageMetrics
from pipeline import PipelineConfig
from pipeline.dual_encoder import DualEncoder
from pipeline.expander import ASTNeighborExpander
from pipeline.index import SnippetIndex
from pipeline.reranker import CrossEncoderReranker

QUESTIONS_DIR = Path("questions")
QUESTIONS_HARD_DIR = Path("questions_hard")
AST_CACHE_DIR = Path("ast_cache")
ABLATION_DIR = Path("ablation_reports")
RERANKER: CrossEncoderReranker


def evaluate_final_modes(
    questions: list[dict],
    dual_encoder: DualEncoder,
    expander: ASTNeighborExpander,
    reranker: CrossEncoderReranker,
    cfg: PipelineConfig,
) -> dict:
    with_ast = StageMetrics("Reranking with AST expansion")
    without_ast = StageMetrics("Reranking without AST expansion")
    no_ast_pairs: list[tuple[str, str]] = []
    with_ast_pairs: list[tuple[str, str]] = []
    no_ast_offsets: list[int] = []
    with_ast_offsets: list[int] = []
    staged_questions: list[dict] = []

    for q in questions:
        query = q["question"]
        stage1 = dual_encoder.search(query, cfg.top_n)
        expanded = expander.expand(stage1)
        no_ast_snippets = [s for s, _ in stage1]
        with_ast_snippets = [s for s, _ in expanded]

        staged_questions.append(
            {
                "query": query,
                "gt_file": q["file"],
                "gt_start": q["line_start"],
                "gt_end": q["line_end"],
                "no_ast_snippets": no_ast_snippets,
                "with_ast_snippets": with_ast_snippets,
            }
        )

        no_ast_offsets.append(len(no_ast_pairs))
        with_ast_offsets.append(len(with_ast_pairs))
        no_ast_pairs.extend(
            (query, reranker._snippet_text(snippet)) for snippet in no_ast_snippets
        )
        with_ast_pairs.extend(
            (query, reranker._snippet_text(snippet)) for snippet in with_ast_snippets
        )

    no_ast_scores = reranker._model.predict(no_ast_pairs, show_progress_bar=False)
    with_ast_scores = reranker._model.predict(with_ast_pairs, show_progress_bar=False)

    for i, q in enumerate(staged_questions):
        no_ast_start = no_ast_offsets[i]
        with_ast_start = with_ast_offsets[i]
        no_ast_end = no_ast_offsets[i + 1] if i + 1 < len(no_ast_offsets) else len(no_ast_pairs)
        with_ast_end = with_ast_offsets[i + 1] if i + 1 < len(with_ast_offsets) else len(with_ast_pairs)

        ranked_no_ast = sorted(
            zip(q["no_ast_snippets"], no_ast_scores[no_ast_start:no_ast_end].tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[: cfg.top_k]
        ranked_with_ast = sorted(
            zip(q["with_ast_snippets"], with_ast_scores[with_ast_start:with_ast_end].tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[: cfg.top_k]

        without_ast.update(
            [s for s, _ in ranked_no_ast],
            q["gt_file"],
            q["gt_start"],
            q["gt_end"],
        )
        with_ast.update(
            [s for s, _ in ranked_with_ast],
            q["gt_file"],
            q["gt_start"],
            q["gt_end"],
        )

    return {
        "rerank_without_ast": without_ast.to_dict(),
        "rerank_with_ast": with_ast.to_dict(),
    }


def run_repo(repo: str, cfg: PipelineConfig) -> None:
    index = SnippetIndex(AST_CACHE_DIR)
    index.load([repo])

    dual_encoder = DualEncoder(cfg.dual_encoder_model, cfg.resolved_device(), cfg.batch_size)
    dual_encoder.index(index.snippets)

    expander = ASTNeighborExpander(index, cfg.hop_depth)

    report: dict = {
        "repo": repo,
        "n_snippets": len(index.snippets),
        "config": {
            "dual_encoder_model": cfg.dual_encoder_model,
            "cross_encoder_model": cfg.cross_encoder_model,
            "top_n": cfg.top_n,
            "hop_depth": cfg.hop_depth,
            "top_k": cfg.top_k,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "questions": {},
    }

    for split, qdir in [("easy", QUESTIONS_DIR), ("hard", QUESTIONS_HARD_DIR)]:
        qfile = qdir / f"{repo}.json"
        if not qfile.exists():
            continue
        questions = json.loads(qfile.read_text())
        print(f"\n--- {repo} / {split} ({len(questions)} questions) ---", flush=True)
        metrics = evaluate_final_modes(questions, dual_encoder, expander, RERANKER, cfg)
        report["questions"][split] = {"n": len(questions), **metrics}
        out = ABLATION_DIR / f"{repo}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"saved {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AST-expansion ablation")
    parser.add_argument("--repos", nargs="+", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--hop-depth", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--dual-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--cross-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    args = parser.parse_args()

    ABLATION_DIR.mkdir(exist_ok=True)

    cfg = PipelineConfig(
        dual_encoder_model=args.dual_model,
        cross_encoder_model=args.cross_model,
        top_n=args.top_n,
        hop_depth=args.hop_depth,
        top_k=args.top_k,
    )

    global RERANKER
    RERANKER = CrossEncoderReranker(cfg.cross_encoder_model, cfg.resolved_device())

    for repo in args.repos:
        run_repo(repo, cfg)


if __name__ == "__main__":
    main()
