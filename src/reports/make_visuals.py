from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ensure_dir


def savefig(path: Path, dpi: int = 220) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_training_curves(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].plot(train_df["episode"], train_df["reward"], color="#1f77b4", lw=1.6, label="train")
    if "reward" in eval_df:
        axes[0].plot(eval_df["episode"], eval_df["reward"], color="#ff7f0e", lw=2.0, label="eval")
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend(frameon=False)

    axes[1].plot(train_df["episode"], train_df["latency_s"], color="#2ca02c", lw=1.6, label="train")
    if "latency_s" in eval_df:
        axes[1].plot(eval_df["episode"], eval_df["latency_s"], color="#d62728", lw=2.0, label="eval")
    axes[1].set_title("Latency")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Seconds")
    axes[1].legend(frameon=False)

    if "feasible_candidates" in train_df:
        axes[2].plot(train_df["episode"], train_df["feasible_candidates"], color="#9467bd", lw=1.6)
        axes[2].set_title("Feasible Candidates")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Count")

    if "loss" in train_df:
        axes[3].plot(train_df["episode"], train_df["loss"], color="#8c564b", lw=1.6, label="loss")
        if "replay_loss" in train_df:
            axes[3].plot(train_df["episode"], train_df["replay_loss"], color="#17becf", lw=1.4, label="replay")
        axes[3].set_title("Optimization")
        axes[3].set_xlabel("Episode")
        axes[3].set_ylabel("Loss")
        axes[3].legend(frameon=False)

    savefig(out_dir / "training_curves.png")


def plot_benchmark_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    df = summary_df.copy()
    df = df.sort_values("reward_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#7f7f7f" if "heuristic" in m or "search" in m or "greedy" in m else "#1f77b4" if "autoreg" in m else "#d62728" if "dros" in m else "#2ca02c" for m in df["method"]]
    ax.barh(df["method"], df["reward_mean"], xerr=df["reward_std"], color=colors, alpha=0.9)
    ax.axvline(df["reward_mean"].max(), color="#111111", ls="--", lw=1, alpha=0.4)
    ax.set_title("Benchmark Reward Comparison")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Method")
    savefig(out_dir / "benchmark_reward_bar.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(df["method"], df["feasible_mean"], color=colors, alpha=0.9)
    ax.set_xlim(0, 1.05)
    ax.set_title("Feasibility Rate")
    ax.set_xlabel("Feasible Rate")
    ax.set_ylabel("Method")
    savefig(out_dir / "benchmark_feasibility_bar.png")


def plot_margin_distribution(rows_df: pd.DataFrame, out_dir: Path) -> None:
    pivot = rows_df.pivot_table(index=["seed", "state_id"], columns="method", values="reward", aggfunc="first")
    heuristic_cols = [c for c in pivot.columns if c != "autoreg_rl_pure"]
    best_heuristic = pivot[heuristic_cols].max(axis=1)
    margins = pivot["autoreg_rl_pure"] - best_heuristic

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(margins, bins=32, color="#1f77b4", alpha=0.9, edgecolor="white")
    ax.axvline(float(margins.mean()), color="#d62728", lw=2, label=f"mean={margins.mean():.4f}")
    ax.axvline(0.0, color="#111111", lw=1, ls="--")
    ax.set_title("Autoreg-RL Margin Distribution")
    ax.set_xlabel("Reward Margin vs Best Heuristic")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    savefig(out_dir / "margin_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    seeds = sorted(rows_df["seed"].unique())
    means = []
    stds = []
    for seed in seeds:
        seed_df = rows_df[rows_df["seed"] == seed]
        seed_pivot = seed_df.pivot_table(index="state_id", columns="method", values="reward", aggfunc="first")
        best_h = seed_pivot[[c for c in seed_pivot.columns if c != "autoreg_rl_pure"]].max(axis=1)
        m = seed_pivot["autoreg_rl_pure"] - best_h
        means.append(float(m.mean()))
        stds.append(float(m.std()))
    ax.errorbar(seeds, means, yerr=stds, fmt="o-", color="#2ca02c", lw=2, capsize=4)
    ax.axhline(0.0, color="#111111", ls="--", lw=1)
    ax.set_title("Per-seed Autoreg-RL Margin")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean Margin")
    savefig(out_dir / "margin_by_seed.png")


def plot_state_scatter(rows_df: pd.DataFrame, out_dir: Path) -> None:
    pivot = rows_df.pivot_table(index=["seed", "state_id"], columns="method", values="reward", aggfunc="first")
    best_heuristic = pivot[[c for c in pivot.columns if c != "autoreg_rl_pure"]].max(axis=1)
    ar = pivot["autoreg_rl_pure"]
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.scatter(best_heuristic, ar, s=22, alpha=0.7, color="#1f77b4", edgecolors="none")
    low = min(float(best_heuristic.min()), float(ar.min()))
    high = max(float(best_heuristic.max()), float(ar.max()))
    ax.plot([low, high], [low, high], color="#111111", ls="--", lw=1)
    ax.set_title("Autoreg-RL vs Best Heuristic")
    ax.set_xlabel("Best Heuristic Reward")
    ax.set_ylabel("Autoreg-RL Reward")
    savefig(out_dir / "autoreg_vs_heuristic_scatter.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="results/autoreg_rl_teacher")
    parser.add_argument("--benchmark-dir", default="results/benchmark_autoreg_1024_3seed")
    parser.add_argument("--out", default="results/visuals_autoreg")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    bench_dir = Path(args.benchmark_dir)
    out_dir = ensure_dir(Path(args.out))

    train_log = train_dir / "train_log.csv"
    eval_log = train_dir / "eval_log.csv"
    benchmark_rows = bench_dir / "benchmark_rows.csv"
    summary_df = pd.read_csv(bench_dir / "benchmark_summary.csv")
    margin = json.loads((bench_dir / "benchmark_margin.json").read_text(encoding="utf-8"))

    generated = []
    if train_log.exists() and eval_log.exists():
        train_df = pd.read_csv(train_log)
        eval_df = pd.read_csv(eval_log)
        plot_training_curves(train_df, eval_df, out_dir)
        generated.append("training_curves.png")
    else:
        print("skipping training_curves.png; train_log.csv/eval_log.csv not found", flush=True)

    plot_benchmark_summary(summary_df, out_dir)
    generated.extend(["benchmark_reward_bar.png", "benchmark_feasibility_bar.png"])
    if benchmark_rows.exists():
        rows_df = pd.read_csv(benchmark_rows)
        plot_margin_distribution(rows_df, out_dir)
        plot_state_scatter(rows_df, out_dir)
        generated.extend(["margin_histogram.png", "margin_by_seed.png", "autoreg_vs_heuristic_scatter.png"])
    else:
        print("skipping margin/scatter plots; benchmark_rows.csv not found", flush=True)

    report_lines = [
        "# Visual Summary",
        "",
        f"- Train dir: `{train_dir.as_posix()}`",
        f"- Benchmark dir: `{bench_dir.as_posix()}`",
        f"- Autoreg-RL margin mean: `{margin['autoreg_rl_pure_margin_mean']:.8f}`",
        f"- Autoreg-RL win/tie rate: `{margin['autoreg_rl_pure_win_or_tie_rate']:.4f}`",
        "",
        "## Generated Files",
        "",
    ]
    report_lines.extend(f"- `{name}`" for name in generated)
    (out_dir / "visual_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote visuals to {out_dir}")


if __name__ == "__main__":
    main()
