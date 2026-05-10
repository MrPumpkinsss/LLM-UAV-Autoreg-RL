from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..autoreg_rl_agent import AutoregRLAgent
from ..baselines import block_beam_strong, block_lns_strong, local_search, random_feasible
from ..benchmark_real_profile import build_real_profile, resolve_real_dir, set_seed
from ..config import ensure_dir, load_config
from ..env import LLMUAVEnv


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def apply_sweep(cfg: dict[str, Any], sweep: str, value: float) -> None:
    if sweep == "bandwidth_mhz":
        cfg["wireless"]["bandwidth_hz"] = float(value) * 1e6
    elif sweep == "energy_scale":
        cfg["uav"]["energy_j_min"] = float(cfg["uav"]["energy_j_min"]) * float(value)
        cfg["uav"]["energy_j_max"] = float(cfg["uav"]["energy_j_max"]) * float(value)
    elif sweep == "area_m":
        cfg["uav"]["area_m"] = float(value)
    elif sweep == "sequence_length":
        cfg["profile"]["sequence_length"] = int(value)
    elif sweep == "snr_threshold":
        cfg["wireless"]["snr_threshold"] = float(value)
    elif sweep == "latency_ref_s":
        cfg["reward"]["latency_ref_s"] = float(value)
    else:
        raise ValueError(f"unsupported sweep: {sweep}")


def evaluate_methods(
    cfg: dict[str, Any],
    real_dir: str | None,
    policy_path: Path,
    seed: int,
    states: int,
    candidates: int,
    device: str,
    methods: set[str],
) -> list[dict[str, Any]]:
    set_seed(seed)
    env_rng = np.random.default_rng(seed)
    agent_rng = np.random.default_rng(seed + 10_000_019)
    profile = build_real_profile(cfg, resolve_real_dir(cfg, real_dir), env_rng)
    env = LLMUAVEnv(cfg, profile, env_rng)
    agent = AutoregRLAgent(env, cfg, device, agent_rng)
    state_dict = torch.load(policy_path, map_location=agent.device)
    agent.policy.load_state_dict(state_dict)

    rows: list[dict[str, Any]] = []
    heur_cfg = cfg.get("heuristics", {})
    max_blocks = int(heur_cfg.get("max_blocks", env.num_uavs))
    beam_width = int(cfg.get("benchmark", {}).get("beam_width", 32))
    lns_steps = int(heur_cfg.get("block_lns_steps", 64))

    for state_id in range(states):
        state = env.sample_state()
        method_actions: dict[str, tuple[np.ndarray, Any, float, int]] = {}

        if "autoreg_rl_pure" in methods:
            started = time.perf_counter()
            selected = agent.select_policy_candidate(
                state,
                candidates=candidates,
                temperature=float(cfg.get("ar_rl", {}).get("eval_temperature", 0.30)),
            )
            method_actions["autoreg_rl_pure"] = (
                selected.action,
                selected.result,
                time.perf_counter() - started,
                selected.feasible_candidates,
            )

        if any(name in methods for name in {"hybrid_heuristic", "block_lns_strong", "block_beam_strong", "random"}):
            random_action = random_feasible(env, state, env_rng)
            random_eval = env.evaluate(state, random_action)
            if "random" in methods:
                method_actions["random"] = (random_action, random_eval, 0.0, -1)

            started = time.perf_counter()
            block_beam_action, block_beam_eval = block_beam_strong(
                env,
                state,
                beam_width=int(heur_cfg.get("block_beam_width", max(beam_width, 256))),
                max_blocks=max_blocks,
            )
            block_beam_runtime = time.perf_counter() - started
            if "block_beam_strong" in methods:
                method_actions["block_beam_strong"] = (block_beam_action, block_beam_eval, block_beam_runtime, -1)

            started = time.perf_counter()
            block_lns_action, block_lns_eval = block_lns_strong(
                env,
                state,
                env_rng,
                initial_actions=[random_action, block_beam_action],
                steps=lns_steps,
                max_blocks=max_blocks,
            )
            block_lns_runtime = time.perf_counter() - started
            if "block_lns_strong" in methods:
                method_actions["block_lns_strong"] = (block_lns_action, block_lns_eval, block_lns_runtime, -1)

            started = time.perf_counter()
            hybrid_action, hybrid_eval = local_search(
                env,
                state,
                env_rng,
                initial_actions=[random_action, block_beam_action, block_lns_action],
                max_passes=int(heur_cfg.get("local_search_passes", 2)),
                random_seed_tries=int(heur_cfg.get("local_search_seed_tries", 64)),
            )
            hybrid_runtime = time.perf_counter() - started
            if "hybrid_heuristic" in methods:
                method_actions["hybrid_heuristic"] = (hybrid_action, hybrid_eval, hybrid_runtime, -1)

        for method, (_action, ev, runtime, feasible_candidates) in method_actions.items():
            rows.append(
                {
                    "seed": seed,
                    "state_id": state_id,
                    "method": method,
                    "reward": float(ev.reward),
                    "cost": float(ev.cost),
                    "feasible": int(ev.feasible),
                    "latency_s": float(ev.latency_s),
                    "ppl_hat": float(ev.ppl_hat),
                    "damage": float(ev.damage),
                    "total_energy_j": float(ev.total_energy_j),
                    "max_mem_ratio": float(ev.max_mem_ratio),
                    "max_energy_ratio": float(ev.max_energy_ratio),
                    "runtime_s": float(runtime),
                    "feasible_candidates": int(feasible_candidates),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]], sweep: str, value: float, label: str) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    output: list[dict[str, Any]] = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        output.append(
            {
                "sweep": sweep,
                "value": value,
                "label": label,
                "method": method,
                "states": len(group),
                "reward_mean": float(np.mean([row["reward"] for row in group])),
                "feasible_mean": float(np.mean([row["feasible"] for row in group])),
                "latency_mean": float(np.mean([row["latency_s"] for row in group])),
                "ppl_mean": float(np.mean([row["ppl_hat"] for row in group])),
                "energy_mean": float(np.mean([row["total_energy_j"] for row in group])),
                "max_energy_ratio_mean": float(np.mean([row["max_energy_ratio"] for row in group])),
                "runtime_mean": float(np.mean([row["runtime_s"] for row in group])),
            }
        )
    return output


def compute_margins(rows: list[dict[str, Any]], sweep: str, value: float, label: str) -> dict[str, Any]:
    by_state: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["state_id"]))
        by_state.setdefault(key, {})[str(row["method"])] = float(row["reward"])
    margins = []
    for methods in by_state.values():
        if "autoreg_rl_pure" not in methods:
            continue
        non_rl = [reward for method, reward in methods.items() if method != "autoreg_rl_pure"]
        if not non_rl:
            continue
        margins.append(methods["autoreg_rl_pure"] - max(non_rl))
    if not margins:
        return {
            "sweep": sweep,
            "value": value,
            "label": label,
            "states": 0,
            "margin_mean": float("nan"),
            "margin_min": float("nan"),
            "win_tie_rate": float("nan"),
        }
    arr = np.asarray(margins, dtype=np.float64)
    return {
        "sweep": sweep,
        "value": value,
        "label": label,
        "states": int(arr.size),
        "margin_mean": float(np.mean(arr)),
        "margin_min": float(np.min(arr)),
        "win_tie_rate": float(np.mean(arr >= -1e-9)),
    }


def plot_sweep(summary_rows: list[dict[str, Any]], margin_rows: list[dict[str, Any]], sweep: str, out_dir: Path) -> None:
    summary = [row for row in summary_rows if row["sweep"] == sweep]
    margins = [row for row in margin_rows if row["sweep"] == sweep]
    if not summary:
        return
    labels = [row["label"] for row in margins]
    x = np.arange(len(labels), dtype=np.float64)
    methods = sorted({row["method"] for row in summary})
    colors = {
        "autoreg_rl_pure": "#1f77b4",
        "hybrid_heuristic": "#2ca02c",
        "block_lns_strong": "#ff7f0e",
        "block_beam_strong": "#9467bd",
        "random": "#7f7f7f",
    }

    for metric, ylabel, filename in [
        ("reward_mean", "Reward", f"{sweep}_reward.png"),
        ("latency_mean", "Latency (s)", f"{sweep}_latency.png"),
        ("ppl_mean", "PPL_hat", f"{sweep}_ppl.png"),
        ("energy_mean", "Energy (J)", f"{sweep}_energy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for method in methods:
            y = []
            for label in labels:
                match = [row for row in summary if row["method"] == method and row["label"] == label]
                y.append(float(match[0][metric]) if match else np.nan)
            ax.plot(x, y, marker="o", lw=2, label=method, color=colors.get(method))
        ax.set_title(f"{sweep}: {ylabel}")
        ax.set_xlabel("Sweep point")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        savefig(out_dir / filename)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, [row["margin_mean"] for row in margins], marker="o", lw=2, color="#d62728", label="mean margin")
    ax.axhline(0.0, color="#111111", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.plot(x, [row["win_tie_rate"] for row in margins], marker="s", lw=2, color="#1f77b4", label="win/tie")
    ax.set_title(f"{sweep}: Autoreg-RL vs best non-RL")
    ax.set_xlabel("Sweep point")
    ax.set_ylabel("Reward margin")
    ax2.set_ylabel("Win/tie rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(alpha=0.25)
    lines, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="best")
    savefig(out_dir / f"{sweep}_margin.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_calibrated.yaml")
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--out", default="results/qwen3_0p6b/sweep_figures")
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--seeds", default="91,92")
    parser.add_argument("--autoreg-candidates", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--methods",
        default="autoreg_rl_pure,hybrid_heuristic,block_lns_strong,block_beam_strong",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    bench_cfg = base_cfg.get("benchmark", {})
    policy_path = Path(
        args.policy
        or bench_cfg.get("policy")
        or Path(base_cfg.get("ar_rl", {}).get("result_dir", "results/qwen3_0p6b/autoreg_rl")) / "autoreg_policy_best.pt"
    )
    real_dir = args.real_dir or bench_cfg.get("real_dir")
    device = str(args.device or bench_cfg.get("device", base_cfg.get("ar_rl", {}).get("device", "cuda")))
    seeds = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    methods = {item.strip() for item in str(args.methods).split(",") if item.strip()}
    out_dir = ensure_dir(args.out)

    sweep_specs = {
        "bandwidth_mhz": [0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
        "energy_scale": [0.70, 0.85, 1.00, 1.15, 1.30],
        "area_m": [500.0, 750.0, 1000.0, 1250.0, 1500.0],
        "sequence_length": [64.0, 96.0, 128.0, 160.0, 192.0],
        "snr_threshold": [5.0, 8.0, 10.0, 12.0, 16.0],
        "latency_ref_s": [3.0, 4.5, 6.0, 7.5, 9.0],
    }

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []

    for sweep, values in sweep_specs.items():
        for value in values:
            cfg = copy.deepcopy(base_cfg)
            apply_sweep(cfg, sweep, value)
            label = f"{value:g}"
            point_rows: list[dict[str, Any]] = []
            for seed in seeds:
                point_rows.extend(
                    evaluate_methods(
                        cfg,
                        real_dir,
                        policy_path,
                        seed,
                        int(args.states),
                        int(args.autoreg_candidates),
                        device,
                        methods,
                    )
                )
            for row in point_rows:
                row["sweep"] = sweep
                row["value"] = float(value)
                row["label"] = label
            all_rows.extend(point_rows)
            summary_rows.extend(summarize(point_rows, sweep, float(value), label))
            margin_rows.append(compute_margins(point_rows, sweep, float(value), label))
            print(f"finished {sweep}={label} rows={len(point_rows)}", flush=True)

    write_csv(out_dir / "sweep_rows.csv", all_rows)
    write_csv(out_dir / "sweep_summary.csv", summary_rows)
    write_csv(out_dir / "sweep_margins.csv", margin_rows)

    for sweep in sweep_specs:
        plot_sweep(summary_rows, margin_rows, sweep, out_dir)

    report = [
        "# Parameter Sweep Figures",
        "",
        f"Config: `{args.config}`",
        f"Policy: `{policy_path.as_posix()}`",
        f"Real profile: `{resolve_real_dir(base_cfg, real_dir).as_posix()}`",
        f"UAVs: `{base_cfg['uav']['num_uavs']}`",
        f"States per point: `{len(seeds) * int(args.states)}` (`{args.seeds}` x `{args.states}`)",
        f"Autoreg candidates: `{int(args.autoreg_candidates)}`",
        "",
        "## Sweeps",
        "",
    ]
    for sweep, values in sweep_specs.items():
        report.append(f"### {sweep}")
        report.append("")
        report.append("| value | RL reward | best non-RL reward | margin | win/tie | RL latency | RL PPL_hat | RL energy |")
        report.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for value in values:
            label = f"{value:g}"
            rl = [row for row in summary_rows if row["sweep"] == sweep and row["label"] == label and row["method"] == "autoreg_rl_pure"][0]
            non_rl = [
                row
                for row in summary_rows
                if row["sweep"] == sweep and row["label"] == label and row["method"] != "autoreg_rl_pure"
            ]
            best = max(non_rl, key=lambda row: row["reward_mean"])
            margin = [row for row in margin_rows if row["sweep"] == sweep and row["label"] == label][0]
            report.append(
                f"| {label} | {rl['reward_mean']:.6f} | {best['reward_mean']:.6f} ({best['method']}) | "
                f"{margin['margin_mean']:.6f} | {margin['win_tie_rate']:.4f} | "
                f"{rl['latency_mean']:.4f} | {rl['ppl_mean']:.4f} | {rl['energy_mean']:.2f} |"
            )
        report.extend(
            [
                "",
                f"![{sweep} reward]({sweep}_reward.png)",
                "",
                f"![{sweep} margin]({sweep}_margin.png)",
                "",
                f"![{sweep} latency]({sweep}_latency.png)",
                "",
                f"![{sweep} PPL]({sweep}_ppl.png)",
                "",
                f"![{sweep} energy]({sweep}_energy.png)",
                "",
            ]
        )
    (out_dir / "sweep_report.md").write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'sweep_report.md'}")


if __name__ == "__main__":
    main()
