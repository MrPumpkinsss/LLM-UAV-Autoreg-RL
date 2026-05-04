from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def estimate_param_bytes(config: Any, dtype: torch.dtype) -> float | None:
    if hasattr(config, "num_parameters"):
        try:
            params = int(config.num_parameters())
            bytes_per_param = 2 if dtype in {torch.float16, torch.bfloat16} else 4
            return float(params * bytes_per_param)
        except Exception:
            return None
    return None


def load_model(model_id: str, cfg: dict[str, Any], dtype: torch.dtype, device_map: Any, trust_remote_code: bool):
    kwargs = dict(
        cache_dir=cfg.get("cache_dir"),
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs), "AutoModelForCausalLM"
    except ValueError as exc:
        message = str(exc)
        if "Unrecognized configuration class" not in message and "not recognized" not in message:
            raise
        return AutoModel.from_pretrained(model_id, **kwargs), "AutoModel"


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal real-LLM deployment smoke test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["metadata", "forward"], default="metadata")
    parser.add_argument("--prompt", default="Distributed LLM inference over UAV links")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_id = str(cfg["model_id"])
    cache_dir = cfg.get("cache_dir")
    dtype = dtype_from_name(cfg.get("dtype", "bfloat16"))
    max_length = int(cfg.get("smoke_max_length", min(int(cfg.get("max_length", 64)), 64)))
    device_map = cfg.get("device_map", {"": cfg.get("device", "cuda")})
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    metadata = {
        "model_id": model_id,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "estimated_param_gib": None,
    }
    est_bytes = estimate_param_bytes(config, dtype)
    if est_bytes is not None:
        metadata["estimated_param_gib"] = est_bytes / 1024**3

    if args.mode == "metadata":
        print(json.dumps(metadata, indent=2), flush=True)
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        before_free, before_total = torch.cuda.mem_get_info()
    else:
        before_free, before_total = 0, 0

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.perf_counter()
    try:
        model, loader = load_model(model_id, cfg, dtype, device_map, trust_remote_code)
        model.eval()
        load_s = time.perf_counter() - started
        encoded = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=max_length)
        first_param_device = next(model.parameters()).device
        encoded = {k: v.to(first_param_device) for k, v in encoded.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with torch.inference_mode():
            out = model(**encoded)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            after_free, after_total = torch.cuda.mem_get_info()
        else:
            after_free, after_total = 0, 0
        metadata.update(
            {
                "loaded": True,
                "loader": loader,
                "load_s": load_s,
                "forward_s": time.perf_counter() - forward_start,
                "output_shape": list(getattr(out, "logits", getattr(out, "last_hidden_state", None)).shape),
                "cuda_total_gib": before_total / 1024**3 if before_total else None,
                "cuda_free_before_gib": before_free / 1024**3 if before_free else None,
                "cuda_free_after_gib": after_free / 1024**3 if after_free else None,
            }
        )
        print(json.dumps(metadata, indent=2), flush=True)
    except Exception as exc:
        metadata.update({"loaded": False, "error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps(metadata, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
