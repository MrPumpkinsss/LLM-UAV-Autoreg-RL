from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ensure_dir
from .real_llm_profile import (
    compute_ppl,
    default_texts,
    device_map_from_config,
    dtype_from_name,
    load_yaml,
    set_seed,
)


@dataclass
class LayerGammaSummary:
    model_id: str
    num_layers: int
    clean_ppl: float
    max_length: int
    batch_size: int
    sample_count: int
    drop_rates: list[float]
    corruption_trials: int
    fitted_gamma_sum: float
    fitted_r2: float
    fitted_rmse_log_ratio: float
    layer_gamma_mean: float
    layer_gamma_min: float
    layer_gamma_max: float


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("could not locate decoder layers on model")


def attach_layer_corruption(model: torch.nn.Module, layer_idx: int, drop_rate: float, seed: int):
    layers = get_decoder_layers(model)
    target_layer = layers[int(layer_idx)]
    generator = torch.Generator(device=next(model.parameters()).device)
    generator.manual_seed(int(seed))

    def hook(_module, _inputs, output):
        if drop_rate <= 0:
            return output
        if isinstance(output, tuple):
            hidden_states = output[0]
            mask = torch.rand(hidden_states.shape[:-1], device=hidden_states.device, generator=generator) >= drop_rate
            noisy_hidden = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
            return (noisy_hidden,) + output[1:]
        mask = torch.rand(output.shape[:-1], device=output.device, generator=generator) >= drop_rate
        return output * mask.unsqueeze(-1).to(output.dtype)

    return target_layer.register_forward_hook(hook)


def fit_layer_gamma(drop_rates: list[float], ppls: list[float], ppl_ref: float) -> tuple[float, float, float]:
    xs = np.asarray(drop_rates, dtype=np.float64)
    ys = np.log(np.maximum(np.asarray(ppls, dtype=np.float64), 1e-12) / max(float(ppl_ref), 1e-12))
    mask = xs > 0
    if np.sum(mask) == 0:
        return 0.0, 0.0, 0.0
    denom = float(np.dot(xs[mask], xs[mask]))
    if denom <= 0:
        return 0.0, 0.0, 0.0
    gamma = float(max(0.0, np.dot(xs[mask], ys[mask]) / denom))
    pred = gamma * xs[mask]
    y = ys[mask]
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return gamma, r2, rmse


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate real layer-wise PPL sensitivity for Qwen3-0.6B.")
    parser.add_argument("--config", default="configs/real_llm.yaml")
    parser.add_argument("--out-dir", default="results/qwen3_0p6b_real_profile")
    parser.add_argument("--layer-drop-rates", default="0.0,0.005,0.01,0.02,0.05,0.08")
    parser.add_argument("--corruption-trials", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-clean-ppl", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    set_seed(int(cfg["seed"]))

    out_dir = ensure_dir(args.out_dir)
    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg["dtype"])
    model_id = str(cfg["model_id"])
    max_length = int(args.max_length if args.max_length is not None else cfg["max_length"])
    batch_size = int(args.batch_size if args.batch_size is not None else cfg["batch_size"])
    drop_rates = [float(x) for x in str(args.layer_drop_rates).split(",") if str(x).strip()]

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cfg.get("cache_dir"), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cfg.get("cache_dir"),
        torch_dtype=dtype,
        device_map=device_map_from_config(cfg, device),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    layers = get_decoder_layers(model)
    num_layers = len(layers)
    boundary_layers = num_layers - 1 if int(args.max_layers) <= 0 else min(int(args.max_layers), num_layers - 1)
    texts = default_texts(cfg)
    if not args.skip_clean_ppl:
        clean_ppl = compute_ppl(model, tokenizer, texts, device, max_length, batch_size)
    else:
        summary_path = out_dir / "real_profile_summary.json"
        if not summary_path.exists():
            summary_path = Path(cfg.get("output_dir", "results/qwen3_0p6b_real_profile")) / "real_profile_summary.json"
        clean_ppl = float(json.loads(summary_path.read_text(encoding="utf-8"))["ppl_ref"])

    rows: list[dict[str, Any]] = []
    layer_gammas: list[float] = []
    layer_r2s: list[float] = []
    layer_rmses: list[float] = []

    for layer_idx in range(boundary_layers):
        layer_ppls = []
        for drop in drop_rates:
            if drop <= 0.0:
                trial_ppls = [float(clean_ppl)] * int(args.corruption_trials)
            else:
                trial_ppls = []
                for trial in range(int(args.corruption_trials)):
                    seed = int(cfg["seed"]) + layer_idx * 10000 + int(drop * 100000) + trial
                    handle = attach_layer_corruption(model, layer_idx, drop, seed)
                    try:
                        trial_ppls.append(compute_ppl(model, tokenizer, texts, device, max_length, batch_size))
                    finally:
                        handle.remove()
            layer_ppls.append(
                {
                    "layer": int(layer_idx),
                    "drop_rate": float(drop),
                    "ppl_mean": float(np.mean(trial_ppls)),
                    "ppl_std": float(np.std(trial_ppls)),
                    "trials": [float(v) for v in trial_ppls],
                }
            )
        gamma, r2, rmse = fit_layer_gamma(
            [float(x["drop_rate"]) for x in layer_ppls],
            [float(x["ppl_mean"]) for x in layer_ppls],
            float(clean_ppl),
        )
        layer_gamma = float(gamma)
        layer_gammas.append(layer_gamma)
        layer_r2s.append(float(r2))
        layer_rmses.append(float(rmse))
        rows.append(
            {
                "layer": int(layer_idx),
                "gamma": layer_gamma,
                "r2": float(r2),
                "rmse_log_ratio": float(rmse),
                "clean_ppl": float(clean_ppl),
                "drop_curve": layer_ppls,
            }
        )

    gamma_arr = np.asarray(layer_gammas, dtype=np.float64)
    summary = LayerGammaSummary(
        model_id=model_id,
        num_layers=int(num_layers),
        clean_ppl=float(clean_ppl),
        max_length=int(max_length),
        batch_size=int(batch_size),
        sample_count=int(len(texts)),
        drop_rates=drop_rates,
        corruption_trials=int(args.corruption_trials),
        fitted_gamma_sum=float(np.sum(gamma_arr)),
        fitted_r2=float(np.mean(layer_r2s)) if layer_r2s else 0.0,
        fitted_rmse_log_ratio=float(np.mean(layer_rmses)) if layer_rmses else 0.0,
        layer_gamma_mean=float(np.mean(gamma_arr)) if gamma_arr.size else 0.0,
        layer_gamma_min=float(np.min(gamma_arr)) if gamma_arr.size else 0.0,
        layer_gamma_max=float(np.max(gamma_arr)) if gamma_arr.size else 0.0,
    )

    np.save(out_dir / "layer_ppl_gamma.npy", gamma_arr)
    (out_dir / "layer_ppl_curve.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "layer_ppl_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    flat_rows = []
    for row in rows:
        for point in row["drop_curve"]:
            flat_rows.append(
                {
                    "layer": int(row["layer"]),
                    "gamma": float(row["gamma"]),
                    "r2": float(row["r2"]),
                    "rmse_log_ratio": float(row["rmse_log_ratio"]),
                    "drop_rate": float(point["drop_rate"]),
                    "ppl_mean": float(point["ppl_mean"]),
                    "ppl_std": float(point["ppl_std"]),
                }
            )
    write_csv(out_dir / "layer_ppl_curve.csv", flat_rows)
    top = sorted(rows, key=lambda item: float(item["gamma"]), reverse=True)[:10]
    lines = [
        "# Layer-Wise PPL Calibration",
        "",
        f"Model: `{model_id}`",
        f"Clean PPL: `{summary.clean_ppl:.6f}`",
        f"Boundary layers: `{len(rows)}`",
        f"Layer gamma sum: `{summary.fitted_gamma_sum:.6f}`",
        f"Mean layer R2: `{summary.fitted_r2:.6f}`",
        f"Mean layer RMSE log-ratio: `{summary.fitted_rmse_log_ratio:.6f}`",
        "",
        "| layer | gamma | R2 | RMSE log-ratio |",
        "|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| {int(row['layer'])} | {float(row['gamma']):.6f} | "
            f"{float(row['r2']):.6f} | {float(row['rmse_log_ratio']):.6f} |"
        )
    (out_dir / "layer_ppl_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
