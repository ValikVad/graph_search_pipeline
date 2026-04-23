from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_counts(full_reports_dir: Path) -> list[tuple[str, int]]:
    counts: list[tuple[str, int]] = []
    for path in sorted(full_reports_dir.glob("*.json")):
        data = json.loads(path.read_text())
        counts.append((data["repo"], int(data["n_snippets"])))
    if not counts:
        raise ValueError(f"No reports found in {full_reports_dir}")
    return sorted(counts, key=lambda item: item[1])


def plot_counts(counts: list[tuple[str, int]], out_path: Path) -> None:
    repos = [repo for repo, _ in counts]
    values = [value for _, value in counts]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(repos, values, color="#4C78A8", edgecolor="#1F2937", linewidth=0.8)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.01,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Число сниппетов по репозиториям", fontsize=18)
    ax.set_ylabel("Число сниппетов", fontsize=12)
    ax.set_xlabel("Репозиторий", fontsize=12)
    ax.set_xticks(range(len(repos)))
    ax.set_xticklabels(repos, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.12)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить график числа сниппетов по репозиториям")
    parser.add_argument("--full-reports", type=Path, default=Path("full_reports"))
    parser.add_argument("--out", type=Path, default=Path("figures/repo_snippet_counts.png"))
    args = parser.parse_args()

    counts = load_counts(args.full_reports)
    plot_counts(counts, args.out)
    print(f"Saved snippet count chart to {args.out}")


if __name__ == "__main__":
    main()
