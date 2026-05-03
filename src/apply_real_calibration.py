from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--real-summary", default="results/qwen3_0p6b_real_profile/real_profile_summary.json")
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
    cfg["output"]["result_dir"] = "results/dros_qwen3_0p6b_calibrated"

    with Path(args.out).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
