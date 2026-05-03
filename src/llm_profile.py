from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def build_qwen3_0p6b_profile(cfg: dict, rng: np.random.Generator) -> LLMProfile:
    """Build a lightweight Qwen3-0.6B per-layer profile.

    The profile is derived from the public Qwen3-0.6B config and uses only
    architecture-level metadata. It does not load model weights.
    """

    n_layers = int(cfg["num_layers"])
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
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
    )
