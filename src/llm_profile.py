from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LLMProfile:
    model_name: str
    num_layers: int
    mem_bytes: np.ndarray
    compute_cycles: np.ndarray
    activation_bytes: np.ndarray
    importance: np.ndarray
    ppl_ref: float
    ppl_gamma: float
    ppl_surrogate: dict[str, Any] | None = None


def load_ppl_surrogate(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    surrogate: dict[str, Any] = {key: data[key] for key in data.files}
    surrogate["type"] = str(np.asarray(surrogate["type"]).item())
    if "input_scale" in surrogate:
        surrogate["input_scale"] = float(np.asarray(surrogate["input_scale"]).item())
    if "max_calibrated_residual" in surrogate:
        surrogate["max_calibrated_residual"] = float(np.asarray(surrogate["max_calibrated_residual"]).item())
    if "num_boundaries" in surrogate:
        surrogate["num_boundaries"] = int(np.asarray(surrogate["num_boundaries"]).item())
    return surrogate


def _mlp_predict_contrib_np(surrogate: dict[str, Any], residuals: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    input_scale = float(surrogate.get("input_scale", 1.0))
    max_residual = float(surrogate.get("max_calibrated_residual", 1.0 / max(input_scale, 1e-12)))
    residual_eval = np.minimum(residuals, max_residual)
    n = int(residuals.shape[-1])
    eye = np.eye(n, dtype=np.float64)
    x = np.concatenate([eye, (residual_eval * input_scale).reshape(n, 1)], axis=1)
    hidden_layers = int(np.asarray(surrogate["hidden_layers"]).item())
    h = x
    for idx in range(hidden_layers):
        w = np.asarray(surrogate[f"w{idx}"], dtype=np.float64)
        b = np.asarray(surrogate[f"b{idx}"], dtype=np.float64)
        preact = h @ w.T + b
        h = preact / (1.0 + np.exp(-preact))
    w_out = np.asarray(surrogate["w_out"], dtype=np.float64)
    b_out = np.asarray(surrogate["b_out"], dtype=np.float64)
    raw = (h @ w_out.T + b_out).reshape(-1)
    gamma = np.logaddexp(raw, 0.0)
    return residuals * gamma


def _mlp_torch_cache(surrogate: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    cache_root = surrogate.setdefault("_torch_cache", {})
    dev_index = "cpu" if device.type == "cpu" else str(device.index if device.index is not None else 0)
    key = f"{device.type}:{dev_index}:{str(dtype).replace('torch.', '')}"
    cached = cache_root.get(key)
    if cached is not None:
        return cached

    hidden_layers = int(np.asarray(surrogate["hidden_layers"]).item())
    weights = []
    biases = []
    for idx in range(hidden_layers):
        weights.append(torch.as_tensor(np.asarray(surrogate[f"w{idx}"], dtype=np.float32), dtype=dtype, device=device))
        biases.append(torch.as_tensor(np.asarray(surrogate[f"b{idx}"], dtype=np.float32), dtype=dtype, device=device))
    cached = {
        "hidden_layers": hidden_layers,
        "input_scale": float(surrogate.get("input_scale", 1.0)),
        "max_calibrated_residual": float(
            surrogate.get(
                "max_calibrated_residual",
                1.0 / max(float(surrogate.get("input_scale", 1.0)), 1e-12),
            )
        ),
        "weights": weights,
        "biases": biases,
        "w_out": torch.as_tensor(np.asarray(surrogate["w_out"], dtype=np.float32), dtype=dtype, device=device),
        "b_out": torch.as_tensor(np.asarray(surrogate["b_out"], dtype=np.float32), dtype=dtype, device=device),
    }
    cache_root[key] = cached
    return cached


def ppl_hat_from_residuals(profile: LLMProfile, residuals: np.ndarray, linear_damage: float | None = None) -> float:
    surrogate = profile.ppl_surrogate
    if surrogate and surrogate.get("type") == "layer_onehot_mlp_v1":
        log_ratio = float(np.sum(_mlp_predict_contrib_np(surrogate, residuals)))
    else:
        if linear_damage is None:
            residuals = np.asarray(residuals, dtype=np.float64)
            linear_damage = float(np.sum(profile.importance * residuals))
        log_ratio = float(profile.ppl_gamma) * float(linear_damage)
    return float(profile.ppl_ref * np.exp(np.clip(log_ratio, -60.0, 60.0)))


def ppl_hat_from_residuals_torch(profile: LLMProfile, residuals: torch.Tensor) -> torch.Tensor:
    surrogate = profile.ppl_surrogate
    if surrogate and surrogate.get("type") == "layer_onehot_mlp_v1":
        device = residuals.device
        dtype = residuals.dtype
        n = residuals.shape[-1]
        cache = _mlp_torch_cache(surrogate, device, dtype)
        input_scale = float(cache["input_scale"])
        max_residual = float(cache["max_calibrated_residual"])
        eye = torch.eye(n, dtype=dtype, device=device)
        leading = residuals.reshape(-1, n)
        onehot = eye.unsqueeze(0).expand(leading.shape[0], n, n)
        residual_eval = torch.clamp(leading, max=max_residual)
        residual_feat = (residual_eval * input_scale).unsqueeze(-1)
        h = torch.cat([onehot, residual_feat], dim=-1).reshape(leading.shape[0] * n, n + 1)
        for idx in range(int(cache["hidden_layers"])):
            h = F.silu(h @ cache["weights"][idx].T + cache["biases"][idx])
        gamma = F.softplus((h @ cache["w_out"].T + cache["b_out"]).reshape(leading.shape[0], n))
        log_ratio = torch.sum(leading * gamma, dim=1).reshape(residuals.shape[:-1])
    else:
        importance = torch.as_tensor(profile.importance, dtype=residuals.dtype, device=residuals.device)
        damage = torch.sum(importance * residuals, dim=-1)
        log_ratio = float(profile.ppl_gamma) * damage
    return float(profile.ppl_ref) * torch.exp(torch.clamp(log_ratio, min=-60.0, max=60.0))


def build_arch_profile(cfg: dict, rng: np.random.Generator) -> LLMProfile:
    """Build a lightweight per-layer profile from architecture metadata."""

    n_layers = int(cfg["num_layers"])
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim") or max(hidden // max(int(cfg["num_attention_heads"]), 1), 1))
    dtype_bytes = int(cfg["dtype_bytes"])
    seq_len = int(cfg["sequence_length"])

    q_params = hidden * hidden
    kv_params = 2 * hidden * (kv_heads * head_dim)
    o_params = hidden * hidden
    mlp_params = 3 * hidden * intermediate
    norm_params = 2 * hidden
    params_per_layer = q_params + kv_params + o_params + mlp_params + norm_params

    base_mem = params_per_layer * dtype_bytes
    layer_jitter = rng.normal(1.0, float(cfg["layer_compute_jitter"]), size=n_layers)
    layer_jitter = np.clip(layer_jitter, 0.75, 1.25)
    mem_bytes = base_mem * np.clip(rng.normal(1.0, 0.04, size=n_layers), 0.9, 1.1)

    # Approximate per-layer single-request compute in CPU-equivalent cycles.
    # A small conversion factor keeps simulated UAV inference latencies in a
    # seconds-level range while preserving the architecture's layer variation.
    flops_per_token = 2.0 * params_per_layer
    compute_cycles = flops_per_token * seq_len * 0.055 * layer_jitter

    base_activation = hidden * seq_len * dtype_bytes
    act_jitter = rng.normal(1.0, float(cfg["activation_jitter"]), size=n_layers - 1)
    act_jitter = np.clip(act_jitter, 0.8, 1.2)
    activation_bytes = base_activation * act_jitter

    importance = activation_bytes / np.maximum(np.sum(activation_bytes), 1.0)

    return LLMProfile(
        model_name=str(cfg["model_name"]),
        num_layers=n_layers,
        mem_bytes=mem_bytes.astype(np.float64),
        compute_cycles=compute_cycles.astype(np.float64),
        activation_bytes=activation_bytes.astype(np.float64),
        importance=importance.astype(np.float64),
        ppl_ref=float(cfg["ppl_ref"]),
        ppl_gamma=float(cfg["ppl_gamma"]),
        ppl_surrogate=None,
    )


def build_qwen3_0p6b_profile(cfg: dict, rng: np.random.Generator) -> LLMProfile:
    return build_arch_profile(cfg, rng)


def build_real_calibrated_profile(cfg: dict, real_dir: str | Path, rng: np.random.Generator) -> LLMProfile:
    real_path = Path(real_dir)
    summary = json.loads((real_path / "real_profile_summary.json").read_text(encoding="utf-8"))
    layer_params = np.load(real_path / "layer_params.npy").astype(np.float64)

    profile_cfg = dict(cfg)
    profile_cfg["num_layers"] = int(summary["num_layers"])
    profile_cfg["hidden_size"] = int(summary["hidden_size"])
    profile_cfg["intermediate_size"] = int(summary["intermediate_size"])
    profile_cfg["num_attention_heads"] = int(summary["num_attention_heads"])
    profile_cfg["num_key_value_heads"] = int(summary["num_key_value_heads"])
    if "head_dim" in summary:
        profile_cfg["head_dim"] = int(summary["head_dim"])
    elif "head_dim" not in profile_cfg:
        profile_cfg["head_dim"] = max(
            int(profile_cfg["hidden_size"]) // max(int(profile_cfg["num_attention_heads"]), 1),
            1,
        )
    profile_cfg["ppl_ref"] = float(summary["ppl_ref"])
    profile_cfg["ppl_gamma"] = float(summary["fitted_gamma"])

    layer_summary_path = real_path / "layer_ppl_summary.json"
    layer_summary = json.loads(layer_summary_path.read_text(encoding="utf-8")) if layer_summary_path.exists() else None
    ppl_ref = float(layer_summary["clean_ppl"]) if layer_summary is not None else float(summary["ppl_ref"])

    base = build_arch_profile(profile_cfg, rng)
    dtype_bytes = int(profile_cfg.get("dtype_bytes", 2))
    mem_bytes = layer_params * dtype_bytes
    activation_bytes = base.activation_bytes.copy()
    layer_gamma_path = real_path / "layer_ppl_gamma.npy"
    if layer_gamma_path.exists():
        layer_gamma = np.load(layer_gamma_path).astype(np.float64)
        expected = int(summary["num_layers"]) - 1
        if layer_gamma.shape != (expected,):
            raise ValueError(f"{layer_gamma_path} has shape {layer_gamma.shape}, expected ({expected},)")
        layer_gamma = np.maximum(layer_gamma, 0.0)
        gamma_sum = float(np.sum(layer_gamma))
        if gamma_sum > 1e-12:
            importance = layer_gamma / gamma_sum
            ppl_gamma = gamma_sum
        else:
            importance = activation_bytes / max(float(np.sum(activation_bytes)), 1.0)
            ppl_gamma = float(summary["fitted_gamma"])
    else:
        importance = activation_bytes / max(float(np.sum(activation_bytes)), 1.0)
        ppl_gamma = float(summary["fitted_gamma"])
    ppl_surrogate = None
    surrogate_mode = str(profile_cfg.get("surrogate_mode", "auto")).lower()
    surrogate_path = real_path / "ppl_surrogate_mlp.npz"
    if surrogate_mode in {"auto", "mlp", "layer_mlp"} and surrogate_path.exists():
        ppl_surrogate = load_ppl_surrogate(surrogate_path)
        expected = int(summary["num_layers"]) - 1
        got = int(ppl_surrogate.get("num_boundaries", expected))
        if got != expected:
            raise ValueError(f"{surrogate_path} has num_boundaries={got}, expected {expected}")
    cycles_scale = mem_bytes / max(float(np.mean(mem_bytes)), 1.0)
    compute_cycles = float(np.mean(base.compute_cycles)) * cycles_scale

    return LLMProfile(
        model_name=str(summary["model_id"]),
        num_layers=int(summary["num_layers"]),
        mem_bytes=mem_bytes.astype(np.float64),
        compute_cycles=compute_cycles.astype(np.float64),
        activation_bytes=activation_bytes.astype(np.float64),
        importance=importance.astype(np.float64),
        ppl_ref=ppl_ref,
        ppl_gamma=ppl_gamma,
        ppl_surrogate=ppl_surrogate,
    )


def build_qwen3_0p6b_real_profile(cfg: dict, real_dir: str | Path, rng: np.random.Generator) -> LLMProfile:
    return build_real_calibrated_profile(cfg, real_dir, rng)
