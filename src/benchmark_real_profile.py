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
from .llm_profile import LLMProfile, build_real_calibrated_profile


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


def build_real_profile(cfg: dict, real_dir: Path, rng: np.random.Generator) -> LLMProfile:
    return build_real_calibrated_profile(cfg["profile"], real_dir, rng)


def resolve_real_dir(cfg: dict, cli_real_dir: str | None) -> Path:
    if cli_real_dir:
        return Path(cli_real_dir)
    configured = cfg.get("profile", {}).get("real_profile_dir")
    if configured:
        return Path(str(configured))
    return Path("results/real_profile")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_calibrated.yaml")
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--states", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--anneal-steps", type=int, default=None)
    parser.add_argument("--autoreg-candidates", type=int, default=None)
    parser.add_argument("--autoreg-refine-steps", type=int, default=None)
    parser.add_argument("--projection-mode", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--candidate-mode", default=None)
    parser.add_argument("--beam-temperature", type=float, default=None)
    parser.add_argument("--candidate-overgenerate", type=int, default=None)
    parser.add_argument("--candidate-beam-count", type=int, default=None)
    parser.add_argument("--candidate-min-hamming", type=int, default=None)
    parser.add_argument("--retransmissions", default=None)
    parser.add_argument("--ppl-cost-mode", default=None, choices=["linear", "log", "cap", "capped"])
    parser.add_argument("--ppl-cost-cap", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.retransmissions is not None:
        retransmissions = str(args.retransmissions)
        if retransmissions.lower() in {"inf", "infinity"}:
            cfg.setdefault("wireless", {})["retransmissions"] = retransmissions
        else:
            cfg.setdefault("wireless", {})["retransmissions"] = int(retransmissions)
    if args.ppl_cost_mode is not None:
        cfg.setdefault("reward", {})["ppl_cost_mode"] = args.ppl_cost_mode
    if args.ppl_cost_cap is not None:
        cfg.setdefault("reward", {})["ppl_cost_cap"] = float(args.ppl_cost_cap)
    bench_cfg = dict(cfg.get("benchmark", {}))
    ar_cfg = cfg.setdefault("ar_rl", {})
    if args.autoreg_refine_steps is not None:
        ar_cfg["policy_refine_steps"] = args.autoreg_refine_steps
    elif "autoreg_refine_steps" in bench_cfg:
        ar_cfg["policy_refine_steps"] = int(bench_cfg["autoreg_refine_steps"])
    benchmark_policy_keys = {
        "projection_mode": args.projection_mode,
        "max_blocks": args.max_blocks,
        "candidate_mode": args.candidate_mode,
        "beam_temperature": args.beam_temperature,
        "candidate_overgenerate": args.candidate_overgenerate,
        "candidate_beam_count": args.candidate_beam_count,
        "candidate_min_hamming": args.candidate_min_hamming,
    }
    for key, value in benchmark_policy_keys.items():
        if value is not None:
            ar_cfg[key] = value
        elif key in bench_cfg:
            ar_cfg[key] = bench_cfg[key]

    real_dir_arg = args.real_dir or bench_cfg.get("real_dir")
    policy_path = Path(args.policy or bench_cfg.get("policy") or Path(ar_cfg.get("result_dir", "results/autoreg_rl")) / "autoreg_policy_best.pt")
    states = int(args.states if args.states is not None else bench_cfg.get("states", 64))
    seeds_text = str(args.seeds if args.seeds is not None else bench_cfg.get("seeds", "91,92,93"))
    beam_width = int(args.beam_width if args.beam_width is not None else bench_cfg.get("beam_width", 32))
    anneal_steps = int(args.anneal_steps if args.anneal_steps is not None else bench_cfg.get("anneal_steps", 128))
    autoreg_candidates = int(
        args.autoreg_candidates
        if args.autoreg_candidates is not None
        else bench_cfg.get("autoreg_candidates", ar_cfg.get("eval_candidates", 64))
    )
    device = str(args.device or bench_cfg.get("device", ar_cfg.get("device", "cuda")))
    out_dir = ensure_dir(args.out or bench_cfg.get("out", "results/benchmark_real_profile_k64"))
    rows: list[dict] = []

    for seed_text in seeds_text.split(","):
        seed = int(seed_text.strip())
        set_seed(seed)
        env_rng = np.random.default_rng(seed)
        agent_rng = np.random.default_rng(seed + 10_000_019)
        profile = build_real_profile(cfg, resolve_real_dir(cfg, real_dir_arg), env_rng)
        env = LLMUAVEnv(cfg, profile, env_rng)
        agent = AutoregRLAgent(env, cfg, device, agent_rng)
        state_dict = torch.load(policy_path, map_location=agent.device)
        agent.policy.load_state_dict(state_dict)

        for sid in range(states):
            state = env.sample_state()
            heuristic_methods = evaluate_full_benchmark_timed(
                env,
                state,
                env_rng,
                beam_width=beam_width,
                anneal_steps=anneal_steps,
            )

            started = time.perf_counter()
            selected = agent.select_policy_candidate(
                state,
                candidates=autoreg_candidates,
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
                print(f"seed={seed} states={sid + 1}/{states}", flush=True)

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

    lines = [
        f"# Real {profile.model_name} Profile Benchmark",
        "",
        f"States: `{margin_summary['states']}`",
        f"Real profile directory: `{resolve_real_dir(cfg, real_dir_arg).as_posix()}`",
        "",
        "| method | reward | feasible | latency | PPL | runtime_s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
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
