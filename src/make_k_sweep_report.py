from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path / "benchmark_summary.csv")
    df["source_dir"] = path.as_posix()
    return df


def load_margin(path: Path) -> dict:
    return json.loads((path / "benchmark_margin.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k16-dir", default="results/benchmark_real_profile_k16_blocks_policy_beam_over4_5seed")
    parser.add_argument("--k64-dir", default="results/benchmark_real_profile_k64_blocks_policy_beam_5seed")
    parser.add_argument("--k128-dir", default="results/benchmark_real_profile_k128_blocks_fast_policy_beam_5seed")
    parser.add_argument("--k256-dir", default="results/benchmark_real_profile_k256_blocks_fast_policy_beam_5seed")
    parser.add_argument("--out-dir", default="results/benchmark_real_profile_k_sweep_fast_5seed")
    args = parser.parse_args()

    k_dirs = {
        16: Path(args.k16_dir),
        64: Path(args.k64_dir),
        128: Path(args.k128_dir),
        256: Path(args.k256_dir),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {k: load_summary(path) for k, path in k_dirs.items()}
    margins = {k: load_margin(path) for k, path in k_dirs.items()}

    baseline_methods = [
        "hybrid_heuristic",
        "beam_search",
        "simulated_annealing",
        "local_search",
        "pdp_aware_greedy",
        "latency_greedy",
        "block_balanced",
        "random",
    ]
    baseline = summaries[256][summaries[256]["method"].isin(baseline_methods)].copy()
    baseline["method"] = baseline["method"].map(lambda name: f"baseline:{name}")
    baseline["k"] = 0

    rl_rows = []
    for k, df in summaries.items():
        row = df[df["method"] == "autoreg_rl_pure"].iloc[0].copy()
        row["method"] = f"autoreg_rl_pure_k{k}"
        row["k"] = k
        row["margin_mean"] = margins[k]["autoreg_rl_pure_margin_mean"]
        row["margin_min"] = margins[k]["autoreg_rl_pure_margin_min"]
        row["win_tie_rate"] = margins[k]["autoreg_rl_pure_win_or_tie_rate"]
        row["strict_win_rate"] = margins[k]["autoreg_rl_pure_strict_win_rate"]
        rl_rows.append(row)
    rl_df = pd.DataFrame(rl_rows)
    for col in ["margin_mean", "margin_min", "win_tie_rate", "strict_win_rate"]:
        baseline[col] = pd.NA

    combined = pd.concat([rl_df, baseline], ignore_index=True)
    combined = combined[
        [
            "method",
            "k",
            "reward_mean",
            "reward_std",
            "feasible_mean",
            "latency_mean",
            "ppl_mean",
            "runtime_mean",
            "margin_mean",
            "margin_min",
            "win_tie_rate",
            "strict_win_rate",
            "source_dir",
        ]
    ].sort_values("reward_mean", ascending=False)
    combined.to_csv(out_dir / "k_sweep_summary.csv", index=False)

    meta = {
        "states": margins[16]["states"],
        "seeds": "91,92,93,94,95",
        "states_per_seed": 64,
        "profile": "Qwen3-0.6B real profile",
        "num_uavs": 5,
        "num_layers": 28,
        "policy": "results/autoreg_rl_real_k16_blocks/autoreg_policy_best.pt",
        "fairness": "RL rows use learned-policy candidates only; no baseline heuristic actions are inserted into the RL candidate pool.",
        "k16": margins[16],
        "k64": margins[64],
        "k128": margins[128],
        "k256": margins[256],
    }
    (out_dir / "k_sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# Real-Profile k Sweep Benchmark",
        "",
        "Fairness: all `autoreg_rl_pure_k*` rows use only learned-policy candidates plus generic feasibility projection. "
        "The strong heuristic actions are benchmark baselines only, not RL candidate-pool entries.",
        "",
        f"States: `{meta['states']}` (`{meta['seeds']}` x `{meta['states_per_seed']}`)",
        f"Profile: `{meta['profile']}`",
        f"Policy: `{meta['policy']}`",
        "",
        "| method | reward | feasible | latency | PPL | runtime_s | margin | win/tie |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in combined.iterrows():
        margin = "" if pd.isna(row["margin_mean"]) else f"{row['margin_mean']:.6f}"
        win = "" if pd.isna(row["win_tie_rate"]) else f"{row['win_tie_rate']:.4f}"
        lines.append(
            f"| {row['method']} | {row['reward_mean']:.6f} +/- {row['reward_std']:.6f} | "
            f"{row['feasible_mean']:.4f} | {row['latency_mean']:.4f} | {row['ppl_mean']:.4f} | "
            f"{row['runtime_mean']:.5f} | {margin} | {win} |"
        )

    lines.extend(
        [
            "",
            "## RL Candidate Budget",
            "",
            "| k | mean margin | min margin | win/tie | strict win | runtime_s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in [16, 64, 128, 256]:
        row = rl_df[rl_df["k"] == k].iloc[0]
        lines.append(
            f"| {k} | {margins[k]['autoreg_rl_pure_margin_mean']:.8f} | "
            f"{margins[k]['autoreg_rl_pure_margin_min']:.8f} | "
            f"{margins[k]['autoreg_rl_pure_win_or_tie_rate']:.4f} | "
            f"{margins[k]['autoreg_rl_pure_strict_win_rate']:.4f} | "
            f"{row['runtime_mean']:.5f} |"
        )

    (out_dir / "k_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "k_sweep_report.md").as_posix())


if __name__ == "__main__":
    main()
