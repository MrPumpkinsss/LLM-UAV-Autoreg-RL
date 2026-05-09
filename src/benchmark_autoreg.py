from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .autoreg_rl_agent import AutoregRLAgent
from .baselines import evaluate_full_benchmark_timed
from .config import ensure_dir, load_config
from .env import LLMUAVEnv
from .llm_profile import build_arch_profile


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_calibrated.yaml")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--seeds", default="91,92,93")
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--anneal-steps", type=int, default=128)
    parser.add_argument("--autoreg-candidates", type=int, default=1024)
    parser.add_argument("--autoreg-refine-steps", type=int, default=0)
    parser.add_argument("--projection-mode", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--candidate-mode", default=None)
    parser.add_argument("--beam-temperature", type=float, default=None)
    parser.add_argument("--candidate-overgenerate", type=int, default=None)
    parser.add_argument("--candidate-beam-count", type=int, default=None)
    parser.add_argument("--candidate-min-hamming", type=int, default=None)
    parser.add_argument("--retransmissions", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.retransmissions is not None:
        retransmissions = str(args.retransmissions)
        if retransmissions.lower() in {"inf", "infinity"}:
            cfg.setdefault("wireless", {})["retransmissions"] = retransmissions
        else:
            cfg.setdefault("wireless", {})["retransmissions"] = int(retransmissions)
    cfg["uav"]["num_uavs"] = 5
    cfg.setdefault("ar_rl", {})["policy_refine_steps"] = args.autoreg_refine_steps
    if args.projection_mode is not None:
        cfg.setdefault("ar_rl", {})["projection_mode"] = args.projection_mode
    if args.max_blocks is not None:
        cfg.setdefault("ar_rl", {})["max_blocks"] = args.max_blocks
    if args.candidate_mode is not None:
        cfg.setdefault("ar_rl", {})["candidate_mode"] = args.candidate_mode
    if args.beam_temperature is not None:
        cfg.setdefault("ar_rl", {})["beam_temperature"] = args.beam_temperature
    if args.candidate_overgenerate is not None:
        cfg.setdefault("ar_rl", {})["candidate_overgenerate"] = args.candidate_overgenerate
    if args.candidate_beam_count is not None:
        cfg.setdefault("ar_rl", {})["candidate_beam_count"] = args.candidate_beam_count
    if args.candidate_min_hamming is not None:
        cfg.setdefault("ar_rl", {})["candidate_min_hamming"] = args.candidate_min_hamming
    bench_cfg = cfg.get("benchmark", {})
    ar_cfg = cfg.get("ar_rl", {})
    default_policy = Path(ar_cfg.get("result_dir", "results/qwen3_0p6b/autoreg_rl")) / "autoreg_policy_best.pt"
    policy_path_arg = Path(args.policy or bench_cfg.get("policy") or str(default_policy))
    out_dir = ensure_dir(args.out or bench_cfg.get("out", "results/qwen3_0p6b/benchmark_autoreg"))
    rows: list[dict] = []

    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        set_seed(seed)
        rng = np.random.default_rng(seed)
        profile = build_arch_profile(cfg["profile"], rng)
        env = LLMUAVEnv(cfg, profile, rng)
        agent = AutoregRLAgent(env, cfg, args.device, rng)
        state_dict = torch.load(policy_path_arg, map_location=agent.device)
        agent.policy.load_state_dict(state_dict)

        for sid in range(args.states):
            state = env.sample_state()
            heuristic_methods = evaluate_full_benchmark_timed(
                env,
                state,
                rng,
                beam_width=args.beam_width,
                anneal_steps=args.anneal_steps,
            )

            started = time.perf_counter()
            selected = agent.select_policy_candidate(
                state,
                candidates=args.autoreg_candidates,
                temperature=float(cfg.get("ar_rl", {}).get("eval_temperature", 0.30)),
            )
            rl_runtime = time.perf_counter() - started

            methods: dict[str, tuple[object, object, float, int]] = {
                "autoreg_rl_pure": (selected.action, selected.result, rl_runtime, selected.feasible_candidates),
            }
            for name, (action, ev, runtime) in heuristic_methods.items():
                methods[name] = (action, ev, runtime, -1)

            for name, (_action, ev, runtime, feasible_candidates) in methods.items():
                rows.append(
                    {
                        "seed": seed,
                        "state_id": sid,
                        "method": name,
                        "reward": ev.reward,
                        "cost": ev.cost,
                        "feasible": int(ev.feasible),
                        "latency_s": ev.latency_s,
                        "ppl_hat": ev.ppl_hat,
                        "damage": ev.damage,
                        "total_energy_j": ev.total_energy_j,
                        "max_mem_ratio": ev.max_mem_ratio,
                        "max_energy_ratio": ev.max_energy_ratio,
                        "runtime_s": runtime,
                        "feasible_candidates": feasible_candidates,
                    }
                )
            if (sid + 1) % 25 == 0:
                print(f"seed={seed} states={sid + 1}/{args.states}", flush=True)

    write_csv(out_dir / "benchmark_rows.csv", rows)
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("method")
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            feasible_mean=("feasible", "mean"),
            latency_mean=("latency_s", "mean"),
            ppl_mean=("ppl_hat", "mean"),
            runtime_mean=("runtime_s", "mean"),
        )
        .sort_values("reward_mean", ascending=False)
    )
    summary.to_csv(out_dir / "benchmark_summary.csv")

    pivot = df.pivot_table(index=["seed", "state_id"], columns="method", values="reward", aggfunc="first")
    best_heuristic = pivot[[c for c in pivot.columns if c != "autoreg_rl_pure"]].max(axis=1)
    margins = pivot["autoreg_rl_pure"] - best_heuristic
    margin_summary = {
        "states": int(len(margins)),
        "autoreg_rl_pure_margin_mean": float(margins.mean()),
        "autoreg_rl_pure_margin_min": float(margins.min()),
        "autoreg_rl_pure_win_or_tie_rate": float((margins >= -1e-9).mean()),
        "autoreg_rl_pure_strict_win_rate": float((margins > 1e-9).mean()),
    }
    (out_dir / "benchmark_margin.json").write_text(json.dumps(margin_summary, indent=2), encoding="utf-8")

    lines = ["# Autoregressive RL Benchmark", "", f"States: `{margin_summary['states']}`", ""]
    lines.append("| method | reward | feasible | latency | PPL | runtime_s |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method, row in summary.iterrows():
        lines.append(
            f"| {method} | {row['reward_mean']:.6f} +/- {row['reward_std']:.6f} | "
            f"{row['feasible_mean']:.4f} | {row['latency_mean']:.4f} | {row['ppl_mean']:.4f} | {row['runtime_mean']:.5f} |"
        )
    lines.append("")
    lines.append(
        f"Autoreg-RL-pure margin vs best heuristic: mean `{margin_summary['autoreg_rl_pure_margin_mean']:.8f}`, "
        f"min `{margin_summary['autoreg_rl_pure_margin_min']:.8f}`, "
        f"win/tie rate `{margin_summary['autoreg_rl_pure_win_or_tie_rate']:.4f}`."
    )
    (out_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string())
    print(json.dumps(margin_summary, indent=2))


if __name__ == "__main__":
    main()
