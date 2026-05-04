from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import ensure_dir


@dataclass
class RealProfileSummary:
    model_id: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    dtype: str
    total_params: int
    layer_param_mean: float
    layer_param_min: int
    layer_param_max: int
    forward_latency_ms_mean: float
    ppl_ref: float
    fitted_gamma: float


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def device_map_from_config(cfg: dict[str, Any], device: torch.device) -> Any:
    if "device_map" not in cfg or cfg.get("device_map") is None:
        return {"": str(device)}
    device_map = cfg["device_map"]
    if isinstance(device_map, str):
        return device_map
    return device_map


def model_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    if hasattr(model, "hf_device_map"):
        device_map = getattr(model, "hf_device_map")
        if isinstance(device_map, dict):
            for value in device_map.values():
                if isinstance(value, str) and value not in {"cpu", "disk"}:
                    return torch.device(value)
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("could not locate decoder layers on model")


def count_layer_params(layers: torch.nn.ModuleList) -> list[int]:
    return [sum(p.numel() for p in layer.parameters()) for layer in layers]


def default_texts(cfg: dict[str, Any]) -> list[str]:
    dataset_cfg = cfg.get("dataset")
    if dataset_cfg:
        try:
            from datasets import load_dataset

            ds = load_dataset(dataset_cfg["name"], dataset_cfg.get("config"), split=dataset_cfg.get("split", "validation"))
            min_chars = int(dataset_cfg.get("min_chars", 80))
            texts = [str(x["text"]).strip() for x in ds if len(str(x.get("text", "")).strip()) >= min_chars]
            if texts:
                return texts[: int(cfg["num_texts"])]
        except Exception as exc:
            print(f"dataset load failed, falling back to configured texts: {exc}", flush=True)

    texts = list(cfg.get("texts", []))
    if not texts:
        texts = [
            "Distributed language model inference can be accelerated by placing different layers across edge devices.",
            "Wireless channel reliability changes the expected latency and the quality of model predictions.",
        ]
    needed = int(cfg["num_texts"])
    while len(texts) < needed:
        texts.extend(texts)
    return texts[:needed]


@torch.no_grad()
def compute_ppl(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int = 1,
) -> float:
    total_nll = 0.0
    total_tokens = 0
    batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_device = model_input_device(model, device)
        input_ids = encoded["input_ids"].to(input_device)
        attention_mask = encoded["attention_mask"].to(input_device)
        if input_ids.shape[1] < 2:
            continue
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        shifted_mask = attention_mask[:, 1:].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        total_nll += float(((-token_log_probs) * shifted_mask).sum().detach().cpu())
        total_tokens += int(shifted_mask.sum().detach().cpu())
    if total_tokens <= 0:
        return float("nan")
    return float(math.exp(total_nll / total_tokens))


def attach_embedding_corruption(model: torch.nn.Module, drop_rate: float, seed: int):
    embed = model.get_input_embeddings()
    generator = torch.Generator(device=embed.weight.device)
    generator.manual_seed(seed)

    def hook(_module, _inputs, output):
        if drop_rate <= 0:
            return output
        mask = torch.rand(output.shape[:-1], device=output.device, generator=generator) >= drop_rate
        return output * mask.unsqueeze(-1).to(output.dtype)

    return embed.register_forward_hook(hook)


@torch.no_grad()
def measure_forward_latency(model: torch.nn.Module, tokenizer: Any, text: str, device: torch.device, max_length: int) -> float:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(model_input_device(model, device))
    for _ in range(3):
        _ = model(input_ids=input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(10):
        started = time.perf_counter()
        _ = model(input_ids=input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
    return float(np.mean(times))


def fit_gamma(drop_rates: list[float], ppls: list[float], ppl_ref: float) -> tuple[float, float, float]:
    xs = np.asarray(drop_rates, dtype=np.float64)
    ys = np.log(np.maximum(np.asarray(ppls, dtype=np.float64), 1e-9) / max(ppl_ref, 1e-9))
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
    return gamma, float(r2), rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_llm.yaml")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    out_dir = ensure_dir(cfg["output_dir"])
    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(cfg["dtype"])
    model_id = str(cfg["model_id"])

    config = AutoConfig.from_pretrained(model_id, cache_dir=cfg.get("cache_dir"), trust_remote_code=True)
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
    layer_params = count_layer_params(layers)
    total_params = sum(p.numel() for p in model.parameters())
    texts = default_texts(cfg)

    latency_ms = measure_forward_latency(model, tokenizer, texts[0], device, int(cfg["max_length"]))
    ppl_ref = compute_ppl(model, tokenizer, texts, device, int(cfg["max_length"]), int(cfg["batch_size"]))

    drop_rates = [float(x) for x in cfg["drop_rates"]]
    rows = []
    for drop in drop_rates:
        ppls = []
        for trial in range(int(cfg["corruption_trials"])):
            handle = attach_embedding_corruption(model, drop, int(cfg["seed"]) + trial + int(drop * 10000))
            try:
                ppls.append(compute_ppl(model, tokenizer, texts, device, int(cfg["max_length"]), int(cfg["batch_size"])))
            finally:
                handle.remove()
        rows.append({"drop_rate": drop, "ppl_mean": float(np.mean(ppls)), "ppl_std": float(np.std(ppls)), "trials": ppls})

    gamma, fit_r2, fit_rmse = fit_gamma([r["drop_rate"] for r in rows], [r["ppl_mean"] for r in rows], ppl_ref)
    summary = RealProfileSummary(
        model_id=model_id,
        num_layers=int(getattr(config, "num_hidden_layers")),
        hidden_size=int(getattr(config, "hidden_size")),
        intermediate_size=int(getattr(config, "intermediate_size")),
        num_attention_heads=int(getattr(config, "num_attention_heads")),
        num_key_value_heads=int(getattr(config, "num_key_value_heads")),
        dtype=str(dtype).replace("torch.", ""),
        total_params=int(total_params),
        layer_param_mean=float(np.mean(layer_params)),
        layer_param_min=int(np.min(layer_params)),
        layer_param_max=int(np.max(layer_params)),
        forward_latency_ms_mean=latency_ms,
        ppl_ref=ppl_ref,
        fitted_gamma=gamma,
    )
    summary_dict = asdict(summary)
    summary_dict["surrogate_fit_r2"] = fit_r2
    summary_dict["surrogate_fit_rmse_log_ratio"] = fit_rmse

    (out_dir / "real_profile_summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    (out_dir / "ppl_corruption_curve.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    np.save(out_dir / "layer_params.npy", np.asarray(layer_params, dtype=np.int64))
    print(json.dumps(summary_dict, indent=2), flush=True)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
