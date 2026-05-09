from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ensure_dir


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_log_affine(rows: list[dict[str, Any]], ppl_ref: float) -> tuple[float, float, dict[str, float]]:
    x = np.asarray([math.log(max(float(row["surrogate_ppl"]), 1e-12) / max(ppl_ref, 1e-12)) for row in rows], dtype=np.float64)
    y = np.asarray([math.log(max(float(row["real_ppl_mean"]), 1e-12) / max(ppl_ref, 1e-12)) for row in rows], dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    residual = pred - y
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    metrics = {
        "r2_log_ratio": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "rmse_log_ratio": float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0,
        "mae_log_ratio": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        "mean_relative_ppl_error": float(np.mean(np.abs(np.exp(pred) - np.exp(y)) / np.maximum(np.exp(y), 1e-12))) if residual.size else 0.0,
    }
    return float(coef[1]), float(coef[0]), metrics


def leave_one_state_out(rows: list[dict[str, Any]], ppl_ref: float) -> dict[str, float]:
    state_ids = np.asarray([int(row["state_id"]) for row in rows], dtype=np.int64)
    x_all = np.asarray([math.log(max(float(row["surrogate_ppl"]), 1e-12) / max(ppl_ref, 1e-12)) for row in rows], dtype=np.float64)
    y_all = np.asarray([math.log(max(float(row["real_ppl_mean"]), 1e-12) / max(ppl_ref, 1e-12)) for row in rows], dtype=np.float64)
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for sid in sorted(set(state_ids.tolist())):
        train_mask = state_ids != sid
        test_mask = ~train_mask
        design = np.column_stack([np.ones(int(train_mask.sum())), x_all[train_mask]])
        coef, *_ = np.linalg.lstsq(design, y_all[train_mask], rcond=None)
        pred = np.column_stack([np.ones(int(test_mask.sum())), x_all[test_mask]]) @ coef
        preds.append(pred)
        truths.append(y_all[test_mask])
    pred_all = np.concatenate(preds)
    true_all = np.concatenate(truths)
    residual = pred_all - true_all
    return {
        "loo_r2_log_ratio": float(1.0 - np.sum(residual**2) / np.sum((true_all - np.mean(true_all)) ** 2)) if true_all.size > 0 else 1.0,
        "loo_rmse_log_ratio": float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0,
        "loo_mean_relative_ppl_error": float(np.mean(np.abs(np.exp(pred_all) - np.exp(true_all)) / np.maximum(np.exp(true_all), 1e-12))) if residual.size else 0.0,
    }


def resolve_ppl_ref(real_dir: Path) -> float:
    layer_summary_path = real_dir / "layer_ppl_summary.json"
    if layer_summary_path.exists():
        layer_summary = json.loads(layer_summary_path.read_text(encoding="utf-8"))
        if "clean_ppl" in layer_summary:
            return float(layer_summary["clean_ppl"])
    summary = json.loads((real_dir / "real_profile_summary.json").read_text(encoding="utf-8"))
    return float(summary["ppl_ref"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a compact action-level log-space calibration for real PPL validation.")
    parser.add_argument("--rows", required=True, help="CSV from benchmark_real_action_ppl.py")
    parser.add_argument("--real-dir", default="results/qwen35_4b/qwen35_4b_real_profile_v2", help="Real profile dir for ppl_ref metadata.")
    parser.add_argument("--out", required=True, help="Calibration JSON output path.")
    args = parser.parse_args()

    rows_path = Path(args.rows)
    rows = read_rows(rows_path)
    if not rows:
        raise ValueError(f"no rows found in {rows_path}")

    real_dir = Path(args.real_dir)
    ppl_ref = resolve_ppl_ref(real_dir)
    scale, bias, train_metrics = fit_log_affine(rows, ppl_ref)
    loo_metrics = leave_one_state_out(rows, ppl_ref)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    payload = {
        "type": "log_ratio_affine_v1",
        "ppl_ref": ppl_ref,
        "bias": bias,
        "scale": scale,
        "rows": int(len(rows)),
        "state_count": int(len(set(int(r["state_id"]) for r in rows))),
        "source_rows": rows_path.as_posix(),
        "source_real_dir": real_dir.as_posix(),
        **train_metrics,
        **loo_metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
