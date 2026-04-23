from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(full_reports_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(full_reports_dir.glob("*.json")):
        data = json.loads(path.read_text())
        repo = data["repo"]
        n_snippets = data.get("n_snippets")
        for record in data["records"]:
            enriched = dict(record)
            enriched["repo"] = repo
            enriched["n_snippets"] = n_snippets
            records.append(enriched)
    if not records:
        raise ValueError(f"No reports found in {full_reports_dir}")
    return records


def filter_records(records: list[dict], excluded_repos: set[str]) -> list[dict]:
    return [record for record in records if record["repo"] not in excluded_repos]


def aggregate_by_repo_split(records: list[dict]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "n": 0,
            "n_snippets": None,
            "stage1_found": 0,
            "stage2_found": 0,
            "stage3_no_ast_found": 0,
            "stage3_found": 0,
            "ast_rescues": 0,
            "final_wins": 0,
            "final_losses": 0,
        }
    )

    for record in records:
        key = (record["repo"], record["split"])
        bucket = grouped[key]
        stage1_found = record["stage1_rank"] is not None
        stage2_found = bool(record["stage2_found"])
        stage3_no_ast_found = record["stage3_no_ast_rank"] is not None
        stage3_found = record["stage3_rank"] is not None

        bucket["n"] += 1
        if bucket["n_snippets"] is None:
            bucket["n_snippets"] = record["n_snippets"]
        bucket["stage1_found"] += int(stage1_found)
        bucket["stage2_found"] += int(stage2_found)
        bucket["stage3_no_ast_found"] += int(stage3_no_ast_found)
        bucket["stage3_found"] += int(stage3_found)
        bucket["ast_rescues"] += int((not stage1_found) and stage2_found)
        bucket["final_wins"] += int((not stage3_no_ast_found) and stage3_found)
        bucket["final_losses"] += int(stage3_no_ast_found and (not stage3_found))

    return grouped


def aggregate_by_repo_split_stage1_top10(records: list[dict]) -> dict[tuple[str, str], dict]:
    return aggregate_by_repo_split_stage1_topk(records, 10)


def aggregate_by_repo_split_stage1_topk(
    records: list[dict],
    top_k: int,
) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "n": 0,
            "n_snippets": None,
            "stage1_found": 0,
            "stage3_no_ast_found": 0,
            "stage3_found": 0,
        }
    )

    for record in records:
        key = (record["repo"], record["split"])
        bucket = grouped[key]
        stage1_topk_found = record["stage1_rank"] is not None and record["stage1_rank"] <= top_k
        stage3_no_ast_found = (
            record["stage3_no_ast_rank"] is not None and record["stage3_no_ast_rank"] <= top_k
        )
        stage3_found = record["stage3_rank"] is not None and record["stage3_rank"] <= top_k

        bucket["n"] += 1
        if bucket["n_snippets"] is None:
            bucket["n_snippets"] = record["n_snippets"]
        bucket["stage1_found"] += int(stage1_topk_found)
        bucket["stage3_no_ast_found"] += int(stage3_no_ast_found)
        bucket["stage3_found"] += int(stage3_found)

    return grouped


def plot_grouped_repo_bars(
    repo_split: dict[tuple[str, str], dict],
    split: str,
    out_path: Path,
    title: str,
) -> None:
    repos = sorted(
        (repo for repo, current_split in repo_split if current_split == split),
        key=lambda repo: repo_split[(repo, split)]["n_snippets"],
    )
    labels = [
        "Этап 1\nDual encoder",
        "Этап 3\nRerank без AST",
        "Этап 3\nRerank с AST",
    ]
    metrics = [
        "stage1_found",
        "stage3_no_ast_found",
        "stage3_found",
    ]
    colours = ["#4C78A8", "#54A24B", "#E45756"]
    repo_labels = [
        f"{repo}\n(snippets={repo_split[(repo, split)]['n_snippets']})"
        for repo in repos
    ]

    x = np.arange(len(repos))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(12, len(repos) * 1.5), 8))
    for idx, (label, metric, colour) in enumerate(zip(labels, metrics, colours)):
        values = [repo_split[(repo, split)][metric] for repo in repos]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width=width, label=label, color=colour)
        for bar, value, repo in zip(bars, values, repos):
            baseline = repo_split[(repo, split)]["stage1_found"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.3,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
            if metric in {"stage3_no_ast_found", "stage3_found"}:
                if baseline > 0:
                    gain_pct = ((value - baseline) / baseline) * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max(0.8, value - 1.0),
                        f"{gain_pct:+.0f}%",
                        ha="center",
                        va="top",
                        fontsize=14,
                        color="white",
                        fontweight="bold",
                        rotation=90,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max(0.8, value - 1.0),
                        "н/д",
                        ha="center",
                        va="top",
                        fontsize=14,
                        color="white",
                        fontweight="bold",
                        rotation=90,
                    )

    ax.set_title(title)
    ax.set_ylabel("Число найденных вопросов")
    ax.set_xticks(x)
    ax.set_xticklabels(repo_labels, rotation=30, ha="right")
    ymax = max(repo_split[(repo, split)][metric] for repo in repos for metric in metrics)
    ax.set_ylim(0, ymax + 4)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hard_delta_bars(
    repo_split: dict[tuple[str, str], dict],
    out_path: Path,
    title: str,
    final_label: str = "top-10",
    baseline_label: str = "этапа 1 top-10",
) -> None:
    repos = sorted(
        (repo for repo, split in repo_split if split == "hard"),
        key=lambda repo: repo_split[(repo, "hard")]["n_snippets"],
    )
    repo_labels = [
        f"{repo}\n(snippets={repo_split[(repo, 'hard')]['n_snippets']})"
        for repo in repos
    ]
    has_stage2 = "stage2_found" in next(iter(repo_split.values()))
    if has_stage2:
        left_gain = [
            repo_split[(repo, "hard")]["stage2_found"] - repo_split[(repo, "hard")]["stage1_found"]
            for repo in repos
        ]
        right_gain = [
            repo_split[(repo, "hard")]["stage3_found"] - repo_split[(repo, "hard")]["stage3_no_ast_found"]
            for repo in repos
        ]
        left_label = "Прирост recall от AST (этап 2 - этап 1)"
        right_label = f"Прирост финального {final_label} (с AST - без AST)"
    else:
        left_gain = [
            repo_split[(repo, "hard")]["stage3_no_ast_found"] - repo_split[(repo, "hard")]["stage1_found"]
            for repo in repos
        ]
        right_gain = [
            repo_split[(repo, "hard")]["stage3_found"] - repo_split[(repo, "hard")]["stage1_found"]
            for repo in repos
        ]
        left_label = f"Прирост финального результата без AST относительно {baseline_label}"
        right_label = f"Прирост финального результата с AST относительно {baseline_label}"

    x = np.arange(len(repos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(repos) * 1.4), 6.5))
    bars1 = ax.bar(x - width / 2, left_gain, width=width, label=left_label, color="#F58518")
    bars2 = ax.bar(x + width / 2, right_gain, width=width, label=right_label, color="#E45756")

    for bars in (bars1, bars2):
        for bar in bars:
            value = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.15 if value >= 0 else -0.35),
                str(value),
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Дополнительно найдено вопросов")
    ax.set_xticks(x)
    ax.set_xticklabels(repo_labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rescue_breakdown(
    records: list[dict],
    out_path: Path,
) -> None:
    splits = ["easy", "hard", "overall"]
    labels = [
        "AST rescue\n(этап 1 промах, этап 2 попадание)",
        "Rescue, дошедшие\nдо финального top-10",
        "Rescue, отброшенные\nreranker'ом",
    ]
    colours = ["#F58518", "#54A24B", "#B279A2"]

    values: dict[str, list[int]] = {}
    for split in splits:
        subset = records if split == "overall" else [r for r in records if r["split"] == split]
        rescues = sum((r["stage1_rank"] is None) and bool(r["stage2_found"]) for r in subset)
        kept = sum((r["stage1_rank"] is None) and bool(r["stage2_found"]) and (r["stage3_rank"] is not None) for r in subset)
        dropped = rescues - kept
        values[split] = [rescues, kept, dropped]

    x = np.arange(len(splits))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (label, colour) in enumerate(zip(labels, colours)):
        series = [values[split][idx] for split in splits]
        bars = ax.bar(x + (idx - 1) * width, series, width=width, label=label, color=colour)
        for bar, value in zip(bars, series):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.4,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Разбор rescue-эффекта AST expansion")
    ax.set_ylabel("Число вопросов")
    ax.set_xticks(x)
    ax.set_xticklabels(["Легкие", "Сложные", "Все"])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary(repo_split: dict[tuple[str, str], dict], records: list[dict], out_path: Path) -> None:
    by_repo_split = {
        f"{repo}:{split}": stats
        for (repo, split), stats in sorted(repo_split.items())
    }

    overall = {
        "n": sum(stats["n"] for stats in repo_split.values()),
        "stage1_found": sum(stats["stage1_found"] for stats in repo_split.values()),
        "stage3_no_ast_found": sum(stats["stage3_no_ast_found"] for stats in repo_split.values()),
        "stage3_found": sum(stats["stage3_found"] for stats in repo_split.values()),
    }
    if all("stage2_found" in stats for stats in repo_split.values()):
        overall["stage2_found"] = sum(stats["stage2_found"] for stats in repo_split.values())
        overall["ast_rescues"] = sum(stats["ast_rescues"] for stats in repo_split.values())
        overall["rescues_kept_to_final"] = sum(
            (r["stage1_rank"] is None) and bool(r["stage2_found"]) and (r["stage3_rank"] is not None)
            for r in records
        )
        overall["rescues_dropped_by_reranker"] = (
            overall["ast_rescues"] - overall["rescues_kept_to_final"]
        )

    out_path.write_text(json.dumps({"overall": overall, "by_repo_split": by_repo_split}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение графиков по full_reports")
    parser.add_argument("--full-reports", type=Path, default=Path("full_reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/full_reports"))
    parser.add_argument(
        "--out-dir-top10",
        type=Path,
        default=Path("figures/full_reports_stage1_top10"),
    )
    parser.add_argument(
        "--out-dir-top3",
        type=Path,
        default=Path("figures/full_reports_stage1_top3"),
    )
    args = parser.parse_args()

    records = load_records(args.full_reports)
    repo_split = aggregate_by_repo_split(records)
    excluded_for_threshold_plots = {"deepseek-ai_DeepSeek-V3"}
    records_threshold = filter_records(records, excluded_for_threshold_plots)
    repo_split_top10 = aggregate_by_repo_split_stage1_top10(records_threshold)
    repo_split_top3 = aggregate_by_repo_split_stage1_topk(records_threshold, 3)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir_top10.mkdir(parents=True, exist_ok=True)
    args.out_dir_top3.mkdir(parents=True, exist_ok=True)

    common_title = "Попадание в top-10 на разных пайплайнах"
    easy_title = f"Легкие вопросы: {common_title}"
    hard_title = f"Сложные вопросы: {common_title}"

    plot_grouped_repo_bars(
        repo_split,
        "easy",
        args.out_dir / "repo_bars_easy.png",
        easy_title,
    )
    plot_grouped_repo_bars(
        repo_split,
        "hard",
        args.out_dir / "repo_bars_hard.png",
        hard_title,
    )
    plot_hard_delta_bars(
        repo_split,
        args.out_dir / "hard_delta_bars.png",
        hard_title,
    )
    plot_rescue_breakdown(records, args.out_dir / "ast_rescue_breakdown.png")
    write_summary(repo_split, records, args.out_dir / "summary.json")

    plot_grouped_repo_bars(
        repo_split_top10,
        "easy",
        args.out_dir_top10 / "repo_bars_easy.png",
        easy_title,
    )
    plot_grouped_repo_bars(
        repo_split_top10,
        "hard",
        args.out_dir_top10 / "repo_bars_hard.png",
        hard_title,
    )
    plot_hard_delta_bars(
        repo_split_top10,
        args.out_dir_top10 / "hard_delta_bars.png",
        hard_title,
        baseline_label="этапа 1 top-10",
    )
    write_summary(repo_split_top10, records_threshold, args.out_dir_top10 / "summary.json")

    common_title_top3 = "Попадание в top-3 на разных пайплайнах"
    easy_title_top3 = f"Легкие вопросы: {common_title_top3}"
    hard_title_top3 = f"Сложные вопросы: {common_title_top3}"

    plot_grouped_repo_bars(
        repo_split_top3,
        "easy",
        args.out_dir_top3 / "repo_bars_easy.png",
        easy_title_top3,
    )
    plot_grouped_repo_bars(
        repo_split_top3,
        "hard",
        args.out_dir_top3 / "repo_bars_hard.png",
        hard_title_top3,
    )
    plot_hard_delta_bars(
        repo_split_top3,
        args.out_dir_top3 / "hard_delta_bars.png",
        hard_title_top3,
        final_label="top-3",
        baseline_label="этапа 1 top-3",
    )
    write_summary(repo_split_top3, records_threshold, args.out_dir_top3 / "summary.json")

    print(f"Saved figures to {args.out_dir}")
    print(f"Saved top-10 baseline figures to {args.out_dir_top10}")
    print(f"Saved top-3 baseline figures to {args.out_dir_top3}")


if __name__ == "__main__":
    main()
