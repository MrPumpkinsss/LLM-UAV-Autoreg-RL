from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .autoreg_rl_agent import AutoregRLAgent
from .baselines import evaluate_full_benchmark_timed, random_feasible
from .benchmark_real_profile import build_real_profile, resolve_real_dir, set_seed
from .config import ensure_dir, load_config
from .env import LLMUAVEnv, SimState
from .real_llm_layer_calibration import attach_layer_corruption
from .real_llm_profile import compute_ppl, default_texts, dtype_from_name, load_yaml


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_action_calibration(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("type", "")).lower() != "log_ratio_affine_v1":
        raise ValueError(f"unsupported action calibration type: {payload.get('type')}")
    return payload


def apply_action_calibration(ppl_hat: float, ppl_ref: float, calibration: dict[str, Any] | None) -> float:
    if calibration is None:
        return float(ppl_hat)
    calibration_ref = float(calibration.get("ppl_ref", ppl_ref))
    base = math.log(max(float(ppl_hat), 1e-12) / max(calibration_ref, 1e-12))
    scale = float(calibration["scale"])
    bias = float(calibration["bias"])
    log_ratio = bias + scale * base
    return calibration_ref * math.exp(max(min(log_ratio, 60.0), -60.0))


def action_residuals(env: LLMUAVEnv, state: SimState, action: np.ndarray) -> dict[int, float]:
    residuals: dict[int, float] = {}
    action = np.asarray(action, dtype=np.int64)
    for layer in range(env.num_layers - 1):
        src = int(action[layer])
        dst = int(action[layer + 1])
        if src == dst:
            continue
        p = float(state.channel.pdp[src, dst])
        _attempts, residual = env._attempts_and_residual(p)
        if residual > 0.0:
            residuals[layer] = float(residual)
    return residuals


def real_action_ppl(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    residuals: dict[int, float],
    seed: int,
    repeats: int,
) -> tuple[float, float, list[float]]:
    if not residuals:
        clean = compute_ppl(model, tokenizer, texts, device, max_length, batch_size)
        return clean, 0.0, [clean] * max(1, int(repeats))

    values: list[float] = []
    for repeat in range(max(1, int(repeats))):
        handles = []
        for layer, residual in residuals.items():
            handles.append(attach_layer_corruption(model, int(layer), float(residual), seed + repeat * 1009 + int(layer) * 9173))
        try:
            values.append(compute_ppl(model, tokenizer, texts, device, max_length, batch_size))
        finally:
            for handle in handles:
                handle.remove()
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)), [float(v) for v in values]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    if float(np.std(xr)) <= 0.0 or float(np.std(yr)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def metric_block(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "rows": 0,
            "mean_abs_ppl_error": float("nan"),
            "max_abs_ppl_error": float("nan"),
            "mean_relative_ppl_error": float("nan"),
            "max_relative_ppl_error": float("nan"),
            "rmse_log_ratio": float("nan"),
            "pearson_ppl": float("nan"),
            "spearman_ppl": float("nan"),
        }
    pred = np.asarray([r["surrogate_ppl"] for r in rows], dtype=np.float64)
    real = np.asarray([r["real_ppl_mean"] for r in rows], dtype=np.float64)
    rel = np.asarray([r["rel_ppl_error"] for r in rows], dtype=np.float64)
    log_err = np.asarray([r["log_ratio_error"] for r in rows], dtype=np.float64)
    if pred.size >= 2 and float(np.std(pred)) > 0.0 and float(np.std(real)) > 0.0:
        pearson = float(np.corrcoef(pred, real)[0, 1])
    else:
        pearson = float("nan")
    return {
        "rows": int(len(rows)),
        "mean_abs_ppl_error": float(np.mean(np.abs(pred - real))),
        "max_abs_ppl_error": float(np.max(np.abs(pred - real))),
        "mean_relative_ppl_error": float(np.mean(rel)),
        "max_relative_ppl_error": float(np.max(rel)),
        "rmse_log_ratio": float(np.sqrt(np.mean(log_err * log_err))),
        "pearson_ppl": pearson,
        "spearman_ppl": spearman_corr(pred, real),
    }


def load_llm(cfg_path: str | Path):
    cfg = load_yaml(cfg_path)
    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg["dtype"])
    model_id = str(cfg["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cfg.get("cache_dir"), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cfg.get("cache_dir"),
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return cfg, model, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate surrogate PPL on real deployment actions.")
    parser.add_argument("--config", default="configs/qwen3_calibrated.yaml")
    parser.add_argument("--llm-config", default="configs/real_llm.yaml")
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--policy", default="results/qwen3_0p6b/autoreg_rl_layer_calibrated_hard_k256/autoreg_policy_best.pt")
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--methods", default="autoreg_rl_pure,hybrid_heuristic,block_lns_strong,block_beam_strong,random")
    parser.add_argument("--autoreg-candidates", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--anneal-steps", type=int, default=128)
    parser.add_argument("--projection-mode", default="blocks_fast")
    parser.add_argument("--max-blocks", type=int, default=5)
    parser.add_argument("--candidate-mode", default="beam")
    parser.add_argument("--beam-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/qwen3_0p6b/real_action_ppl_validation")
    parser.add_argument("--calibration-file", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["uav"]["num_uavs"] = 5
    cfg.setdefault("ar_rl", {})["policy_refine_steps"] = 0
    cfg.setdefault("ar_rl", {})["projection_mode"] = args.projection_mode
    cfg.setdefault("ar_rl", {})["max_blocks"] = int(args.max_blocks)
    cfg.setdefault("ar_rl", {})["candidate_mode"] = args.candidate_mode
    cfg.setdefault("ar_rl", {})["beam_temperature"] = float(args.beam_temperature)
    cfg.setdefault("heuristics", {})["max_blocks"] = int(args.max_blocks)

    llm_cfg, model, tokenizer, llm_device = load_llm(args.llm_config)
    max_length = int(args.max_length if args.max_length is not None else llm_cfg["max_length"])
    batch_size = int(args.batch_size if args.batch_size is not None else llm_cfg["batch_size"])
    texts = default_texts(llm_cfg)
    out_dir = ensure_dir(args.out)
    wanted_methods = {x.strip() for x in str(args.methods).split(",") if x.strip()}
    action_calibration = load_action_calibration(args.calibration_file)

    set_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    profile = build_real_profile(cfg, resolve_real_dir(cfg, args.real_dir), rng)
    env = LLMUAVEnv(cfg, profile, rng)
    agent = AutoregRLAgent(env, cfg, args.device, rng)
    state_dict = torch.load(Path(args.policy), map_location=agent.device)
    agent.policy.load_state_dict(state_dict)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, tuple[int, ...]]] = set()
    for state_id in range(int(args.states)):
        state = env.sample_state()
        method_actions: dict[str, tuple[np.ndarray, Any]] = {}
        selected = agent.select_policy_candidate(
            state,
            candidates=int(args.autoreg_candidates),
            temperature=float(cfg.get("ar_rl", {}).get("eval_temperature", 0.30)),
        )
        method_actions["autoreg_rl_pure"] = (selected.action, selected.result)
        heuristics = evaluate_full_benchmark_timed(
            env,
            state,
            rng,
            beam_width=int(args.beam_width),
            anneal_steps=int(args.anneal_steps),
        )
        for name, (action, ev, _runtime) in heuristics.items():
            method_actions[name] = (action, ev)
        if "random" not in method_actions:
            action = random_feasible(env, state, rng)
            method_actions["random"] = (action, env.evaluate(state, action))

        for method, (action, ev) in method_actions.items():
            if method not in wanted_methods:
                continue
            key = (state_id, method, tuple(int(x) for x in np.asarray(action, dtype=np.int64).tolist()))
            if key in seen:
                continue
            seen.add(key)
            residuals = action_residuals(env, state, action)
            real_ppl_mean, real_ppl_std, real_trials = real_action_ppl(
                model,
                tokenizer,
                texts,
                llm_device,
                max_length,
                batch_size,
                residuals,
                seed=int(args.seed) + state_id * 100000 + len(rows) * 137,
                repeats=int(args.repeats),
            )
            raw_surrogate_ppl = float(ev.ppl_hat)
            surrogate_ppl = apply_action_calibration(raw_surrogate_ppl, float(profile.ppl_ref), action_calibration)
            log_error = float(math.log(max(surrogate_ppl, 1e-12) / max(real_ppl_mean, 1e-12)))
            rows.append(
                {
                    "state_id": int(state_id),
                    "method": method,
                    "raw_surrogate_ppl": raw_surrogate_ppl,
                    "surrogate_ppl": surrogate_ppl,
                    "real_ppl_mean": real_ppl_mean,
                    "real_ppl_std": real_ppl_std,
                    "abs_ppl_error": abs(surrogate_ppl - real_ppl_mean),
                    "rel_ppl_error": abs(surrogate_ppl - real_ppl_mean) / max(real_ppl_mean, 1e-12),
                    "log_ratio_error": log_error,
                    "latency_s": float(ev.latency_s),
                    "reward": float(ev.reward),
                    "damage": float(ev.damage),
                    "transition_count": int(len(residuals)),
                    "residual_sum": float(sum(residuals.values())),
                    "residual_max": float(max(residuals.values()) if residuals else 0.0),
                    "action": " ".join(str(int(x)) for x in np.asarray(action, dtype=np.int64).tolist()),
                    "residuals_json": json.dumps({str(k): float(v) for k, v in residuals.items()}, sort_keys=True),
                    "real_trials_json": json.dumps(real_trials),
                }
            )
        print(f"validated state {state_id + 1}/{args.states}", flush=True)

    write_csv(out_dir / "real_action_ppl_rows.csv", rows)
    by_method = {
        method: metric_block([row for row in rows if str(row["method"]) == method])
        for method in sorted({str(row["method"]) for row in rows})
    }
    competitive_rows = [row for row in rows if str(row["method"]) != "random"]
    metrics = {
        "states": int(args.states),
        "methods": sorted(wanted_methods),
        "real_dir": resolve_real_dir(cfg, args.real_dir).as_posix(),
        "policy": Path(args.policy).as_posix(),
        "calibration_file": Path(args.calibration_file).as_posix() if args.calibration_file else None,
        "all": metric_block(rows),
        "competitive_non_random": metric_block(competitive_rows),
        "by_method": by_method,
    }
    (out_dir / "real_action_ppl_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# Real Action-Level PPL Validation",
        "",
        f"Real profile directory: `{resolve_real_dir(cfg, args.real_dir).as_posix()}`",
        f"Policy: `{Path(args.policy).as_posix()}`",
        f"Action calibration: `{Path(args.calibration_file).as_posix()}`" if args.calibration_file else "Action calibration: none",
        f"Rows: `{metrics['all']['rows']}`",
        "",
        "| scope | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| all | {metrics['all']['rows']} | {metrics['all']['mean_relative_ppl_error']:.6f} | "
        f"{metrics['all']['max_relative_ppl_error']:.6f} | {metrics['all']['rmse_log_ratio']:.6f} | "
        f"{metrics['all']['pearson_ppl']:.6f} | {metrics['all']['spearman_ppl']:.6f} |",
        f"| non-random competitive | {metrics['competitive_non_random']['rows']} | "
        f"{metrics['competitive_non_random']['mean_relative_ppl_error']:.6f} | "
        f"{metrics['competitive_non_random']['max_relative_ppl_error']:.6f} | "
        f"{metrics['competitive_non_random']['rmse_log_ratio']:.6f} | "
        f"{metrics['competitive_non_random']['pearson_ppl']:.6f} | "
        f"{metrics['competitive_non_random']['spearman_ppl']:.6f} |",
        "",
        "| method | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in by_method.items():
        lines.append(
            f"| {method} | {item['rows']} | {item['mean_relative_ppl_error']:.6f} | "
            f"{item['max_relative_ppl_error']:.6f} | {item['rmse_log_ratio']:.6f} | "
            f"{item['pearson_ppl']:.6f} | {item['spearman_ppl']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| method | surrogate PPL | real PPL | rel error | transitions |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['surrogate_ppl']:.6f} | "
            f"{row['real_ppl_mean']:.6f} +/- {row['real_ppl_std']:.6f} | "
            f"{row['rel_ppl_error']:.6f} | {row['transition_count']} |"
        )
    (out_dir / "real_action_ppl_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {out_dir / 'real_action_ppl_report.md'}", flush=True)


if __name__ == "__main__":
    main()
