from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--real-summary", default="results/qwen3_0p6b/qwen3_0p6b_real_profile/real_profile_summary.json")
    parser.add_argument("--out", default="configs/qwen3_calibrated.yaml")
    args = parser.parse_args()

    cfg = load_config(args.base_config)
    summary = json.loads(Path(args.real_summary).read_text(encoding="utf-8"))
    cfg["profile"]["ppl_ref"] = float(summary["ppl_ref"])
    cfg["profile"]["ppl_gamma"] = float(summary["fitted_gamma"])
    cfg["profile"]["num_layers"] = int(summary["num_layers"])
    cfg["profile"]["hidden_size"] = int(summary["hidden_size"])
    cfg["profile"]["intermediate_size"] = int(summary["intermediate_size"])
    cfg["profile"]["num_attention_heads"] = int(summary["num_attention_heads"])
    cfg["profile"]["num_key_value_heads"] = int(summary["num_key_value_heads"])
    if "head_dim" in summary:
        cfg["profile"]["head_dim"] = int(summary["head_dim"])
    model_name = str(summary.get("model_id", cfg["profile"].get("model_name", "llm")))
    slug = model_name.split("/")[-1].replace("-", "_").replace(".", "_").lower()
    family_dir = (
        "qwen3_0p6b"
        if "0_6b" in slug or "0p6b" in slug
        else "qwen35_4b"
        if "3_5_4b" in slug or "35_4b" in slug
        else "gemma4_e4b"
        if "gemma" in slug and "e4b" in slug
        else "cross_model"
    )
    cfg["profile"]["model_name"] = model_name
    cfg["output"]["result_dir"] = f"results/{family_dir}/dros_{slug}_calibrated"

    with Path(args.out).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
