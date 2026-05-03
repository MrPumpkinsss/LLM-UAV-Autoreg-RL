from __future__ import annotations

import argparse
import csv
import json
import itertools
from pathlib import Path

import numpy as np
import torch

from .autoreg_rl_agent import AutoregRLAgent
from .baselines import evaluate_full_benchmark
from .config import ensure_dir, load_config
from .env import LLMUAVEnv
from .llm_profile import LLMProfile, build_qwen3_0p6b_profile


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def truncate_profile(profile: LLMProfile, num_layers: int) -> LLMProfile:
    if num_layers >= profile.num_layers:
        return profile
    if num_layers < 2:
        raise ValueError("num_layers must be at least 2")
    return LLMProfile(
        model_name=f"{profile.model_name}-L{num_layers}",
        num_layers=num_layers,
        mem_bytes=profile.mem_bytes[:num_layers].copy(),
        compute_cycles=profile.compute_cycles[:num_layers].copy(),
        activation_bytes=profile.activation_bytes[: num_layers - 1].copy(),
        importance=profile.importance[: num_layers - 1].copy()
        / max(float(np.sum(profile.importance[: num_layers - 1])), 1e-12),
        ppl_ref=profile.ppl_ref,
        ppl_gamma=profile.ppl_gamma,
    )


def exhaustive_best(env: LLMUAVEnv, state, max_states: int) -> tuple[np.ndarray, object]:
    total = env.num_uavs ** env.num_layers
    if total > max_states:
        raise ValueError(f"exhaustive space too large: {total} > {max_states}")
    best_action = None
    best_eval = None
    for combo in itertools.product(range(env.num_uavs), repeat=env.num_layers):
        action = np.asarray(combo, dtype=np.int64)
        ev = env.evaluate(state, action)
        if ev.feasible and (best_eval is None or ev.reward > best_eval.reward):
            best_action = action.copy()
            best_eval = ev
    if best_action is None:
        # Return best infeasible shaped only for diagnostics.
        for combo in itertools.product(range(env.num_uavs), repeat=env.num_layers):
            action = np.asarray(combo, dtype=np.int64)
            ev = env.evaluate(state, action)
            if best_eval is None or ev.reward > best_eval.reward:
                best_action = action.copy()
                best_eval = ev
    assert best_action is not None and best_eval is not None
    return best_action, best_eval


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
    parser.add_argument("--policy", default="results/autoreg_rl_teacher/autoreg_policy_best.pt")
    parser.add_argument("--out", default="results/exact_optimal_L7_N5")
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--num-uavs", type=int, default=5)
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--autoreg-candidates", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--anneal-steps", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["uav"]["num_uavs"] = args.num_uavs
    cfg["profile"]["num_layers"] = args.num_layers
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = ensure_dir(args.out)

    full_profile = build_qwen3_0p6b_profile(cfg["profile"], rng)
    profile = truncate_profile(full_profile, args.num_layers)
    env = LLMUAVEnv(cfg, profile, rng)

    agent = AutoregRLAgent(env, cfg, args.device, rng)
    policy_path = Path(args.policy)
    policy_loaded = False
    if policy_path.exists():
        state_dict = torch.load(policy_path, map_location=agent.device)
        try:
            agent.policy.load_state_dict(state_dict)
            policy_loaded = True
        except RuntimeError:
            print("policy shape does not match small exact environment; using untrained small policy", flush=True)

    rows = []
    for sid in range(args.states):
        state = env.sample_state()
        opt_action, opt_ev = exhaustive_best(env, state, args.max_states)
        heuristic = evaluate_full_benchmark(env, state, rng, beam_width=args.beam_width, anneal_steps=args.anneal_steps)
        best_h_name, (best_h_action, best_h_ev) = max(heuristic.items(), key=lambda item: item[1][1].reward)
        ar_sel = agent.select_policy_candidate(
            state,
            candidates=args.autoreg_candidates,
            temperature=float(cfg.get("ar_rl", {}).get("eval_temperature", 0.30)),
        )
        for method, ev, winner in [
            ("optimal", opt_ev, "optimal"),
            ("best_heuristic", best_h_ev, best_h_name),
            ("autoreg_rl_pure", ar_sel.result, "autoreg_rl_pure"),
        ]:
            rows.append(
                {
                    "state_id": sid,
                    "method": method,
                    "winner": winner,
                    "reward": ev.reward,
                    "cost": ev.cost,
                    "feasible": int(ev.feasible),
                    "latency_s": ev.latency_s,
                    "ppl_hat": ev.ppl_hat,
                    "optimality_gap": max(0.0, opt_ev.reward - ev.reward),
                }
            )
        if (sid + 1) % 10 == 0:
            print(f"states={sid + 1}/{args.states}", flush=True)

    write_csv(out_dir / "exact_rows.csv", rows)
    import pandas as pd

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("method")
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            feasible_mean=("feasible", "mean"),
            gap_mean=("optimality_gap", "mean"),
            gap_max=("optimality_gap", "max"),
            latency_mean=("latency_s", "mean"),
            ppl_mean=("ppl_hat", "mean"),
        )
        .sort_values("reward_mean", ascending=False)
    )
    summary.to_csv(out_dir / "exact_summary.csv")

    meta = {
        "num_layers": args.num_layers,
        "num_uavs": args.num_uavs,
        "states": args.states,
        "search_space_per_state": int(args.num_uavs ** args.num_layers),
        "policy_loaded": policy_loaded,
    }
    (out_dir / "exact_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# Exact Optimal Comparison",
        "",
        "Exact exhaustive optimal comparison is only tractable for reduced toy instances.",
        f"This run uses `{args.num_layers}` layers and `{args.num_uavs}` UAVs, "
        f"so each state has `{args.num_uavs ** args.num_layers}` assignments.",
        "",
        f"Policy loaded: `{policy_loaded}`",
        "",
        "| method | reward | feasible | gap_mean | gap_max | latency | PPL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary.iterrows():
        lines.append(
            f"| {method} | {row['reward_mean']:.6f} +/- {row['reward_std']:.6f} | "
            f"{row['feasible_mean']:.4f} | {row['gap_mean']:.6f} | {row['gap_max']:.6f} | "
            f"{row['latency_mean']:.4f} | {row['ppl_mean']:.4f} |"
        )
    (out_dir / "exact_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string())
    print(f"wrote {out_dir / 'exact_report.md'}")


if __name__ == "__main__":
    main()
