from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="results/autoreg_rl_teacher")
    parser.add_argument("--benchmark-dir", default="results/benchmark_autoreg_1024_3seed")
    parser.add_argument("--out", default="results/autoreg_rl_strong_benchmark_report.md")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    bench_dir = Path(args.benchmark_dir)
    summary = json.loads((train_dir / "summary.json").read_text(encoding="utf-8"))
    eval_df = pd.read_csv(train_dir / "eval_log.csv")
    bench_summary = pd.read_csv(bench_dir / "benchmark_summary.csv")
    margin = json.loads((bench_dir / "benchmark_margin.json").read_text(encoding="utf-8"))
    best_eval = eval_df.loc[eval_df["reward"].idxmax()].to_dict()

    lines = [
        "# Autoregressive Pure RL Strong Benchmark",
        "",
        "Inference fairness: `autoreg_rl_pure` uses only actions sampled from the autoregressive RL policy plus generic feasibility projection. "
        "No beam, local-search, simulated-annealing, or greedy heuristic action is inserted into its candidate pool.",
        "",
        "Training note: this checkpoint uses teacher warm-start from strong heuristic solutions, followed by RL/self-imitation updates. "
        "This is not a heuristic candidate pool at inference time, but it should be described as teacher-warm-start RL.",
        "",
        "## Training",
        "",
        f"- Train directory: `{train_dir.as_posix()}`",
        f"- Teacher reward mean: `{summary.get('teacher_reward_mean'):.6f}`",
        f"- Best validation episode: `{int(best_eval['episode'])}`",
        f"- Best validation reward: `{best_eval['reward']:.6f}`",
        f"- Best validation feasible rate: `{best_eval['feasible_rate']:.4f}`",
        f"- Checkpoint: `{(train_dir / 'autoreg_policy_best.pt').as_posix()}`",
        "",
        "## Benchmark",
        "",
        f"- Benchmark directory: `{bench_dir.as_posix()}`",
        f"- States: `{margin['states']}`",
        "",
        "| method | reward | feasible | latency | PPL | runtime_s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in bench_summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['reward_mean']:.6f} +/- {row['reward_std']:.6f} | "
            f"{row['feasible_mean']:.4f} | {row['latency_mean']:.4f} | {row['ppl_mean']:.4f} | {row['runtime_mean']:.5f} |"
        )
    mean_margin = float(margin["autoreg_rl_pure_margin_mean"])
    if mean_margin >= 0.0:
        result_line = "- Under this benchmark, autoregressive pure RL exceeds the strongest heuristic on mean reward."
    else:
        result_line = "- Under this benchmark, autoregressive pure RL does not exceed the strongest heuristic on mean reward."
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- `autoreg_rl_pure` margin vs best non-RL heuristic: `{mean_margin:.8f}` mean, "
            f"`{margin['autoreg_rl_pure_margin_min']:.8f}` min.",
            f"- `autoreg_rl_pure` win/tie rate vs best non-RL heuristic: `{margin['autoreg_rl_pure_win_or_tie_rate']:.4f}`.",
            result_line,
            "- `dros_hybrid` remains an upper-reference hybrid method and should not be called pure RL.",
        ]
    )
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
