from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ensure_dir


def fit_gamma(drop_rates: np.ndarray, y_log_ratio: np.ndarray) -> float:
    mask = drop_rates > 0
    denom = float(np.dot(drop_rates[mask], drop_rates[mask]))
    if denom <= 0:
        return 0.0
    return float(max(0.0, np.dot(drop_rates[mask], y_log_ratio[mask]) / denom))


def fit_polynomial_through_origin(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    degree = max(1, int(degree))
    powers = [x ** p for p in range(1, degree + 1)]
    design = np.stack(powers, axis=1)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ coef
    return coef.astype(np.float64), pred.astype(np.float64)


def predict_polynomial(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(x, dtype=np.float64)
    for idx, value in enumerate(coef, start=1):
        pred += float(value) * (x ** idx)
    return pred


def monotone_cumulative_max(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(y, dtype=np.float64))


def metric_block(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    residual = y[mask] - pred[mask]
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return {
        "r2_log_ratio": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "rmse_log_ratio": float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_curve(rows: list[dict], out_dir: Path, label: str) -> None:
    xs = np.asarray([r["drop_rate"] for r in rows], dtype=np.float64)
    y_real = np.asarray([r["real_ppl_mean"] for r in rows], dtype=np.float64)
    y_pred = np.asarray([r["surrogate_ppl"] for r in rows], dtype=np.float64)
    y_std = np.asarray([r["real_ppl_std"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(xs, y_real, yerr=y_std, fmt="o", color="#1f77b4", capsize=4, label="Real model")
    ax.plot(xs, y_pred, color="#d62728", lw=2.0, label=label)
    ax.set_xlabel("Embedding Drop Rate")
    ax.set_ylabel("Perplexity")
    ax.set_title("Surrogate PPL Fit")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "surrogate_ppl_fit.png", dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default="results/real_profile")
    parser.add_argument("--out", default="results/surrogate_benchmark")
    parser.add_argument(
        "--fit",
        default="auto",
        choices=["auto", "exponential", "poly2", "poly3", "piecewise", "monotone_piecewise"],
        help="Fit used for embedding-drop curve benchmarks. Auto keeps exponential for good curves and uses piecewise for noisy sparse curves.",
    )
    parser.add_argument("--write-surrogate", action="store_true", help="Write ppl_surrogate_curve.json into the real profile directory.")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    out_dir = ensure_dir(args.out)
    summary = json.loads((real_dir / "real_profile_summary.json").read_text(encoding="utf-8"))
    curve = json.loads((real_dir / "ppl_corruption_curve.json").read_text(encoding="utf-8"))
    layer_summary_path = real_dir / "layer_ppl_summary.json"
    layer_summary = json.loads(layer_summary_path.read_text(encoding="utf-8")) if layer_summary_path.exists() else None
    ppl_ref = float(summary["ppl_ref"])
    drops = np.asarray([float(item["drop_rate"]) for item in curve], dtype=np.float64)
    real_ppls = np.asarray([float(item["ppl_mean"]) for item in curve], dtype=np.float64)
    y = np.log(np.maximum(real_ppls, 1e-12) / max(ppl_ref, 1e-12))
    mask = drops > 0

    gamma = fit_gamma(drops, y)
    pred_exp = gamma * drops
    exp_metrics = metric_block(y, pred_exp, mask)
    fit = str(args.fit)
    coef: np.ndarray | None = None
    if fit == "auto":
        positive_points = int(np.sum(mask))
        if positive_points <= 4:
            fit = "poly3"
        else:
            fit = "exponential" if exp_metrics["r2_log_ratio"] >= 0.99 else "piecewise"
    if fit == "exponential":
        pred_log = pred_exp
        fit_label = "Exponential surrogate"
    elif fit in {"poly2", "poly3"}:
        degree = 2 if fit == "poly2" else 3
        coef, pred_log = fit_polynomial_through_origin(drops[mask], y[mask], degree)
        pred_log_all = predict_polynomial(drops, coef)
        pred_log_all[~mask] = 0.0
        pred_log = pred_log_all
        fit_label = f"Polynomial degree {degree} surrogate"
    elif fit == "piecewise":
        pred_log = y.copy()
        pred_log[~mask] = 0.0
        fit_label = "Empirical piecewise surrogate"
    elif fit == "monotone_piecewise":
        pred_log = monotone_cumulative_max(y)
        pred_log[~mask] = 0.0
        fit_label = "Monotone empirical piecewise surrogate"
    else:
        raise ValueError(f"unknown fit: {fit}")

    rows = []
    for idx, item in enumerate(curve):
        drop = float(item["drop_rate"])
        real_ppl = float(item["ppl_mean"])
        pred_value = float(pred_log[idx])
        pred_ppl = float(ppl_ref * np.exp(pred_value))
        log_error = float(np.log(max(pred_ppl, 1e-12) / max(real_ppl, 1e-12)))
        rows.append(
            {
                "drop_rate": drop,
                "real_ppl_mean": real_ppl,
                "real_ppl_std": float(item["ppl_std"]),
                "surrogate_ppl": pred_ppl,
                "abs_ppl_error": abs(pred_ppl - real_ppl),
                "rel_ppl_error": abs(pred_ppl - real_ppl) / max(real_ppl, 1e-12),
                "log_ratio_error": log_error,
            }
        )

    abs_errors = np.asarray([r["abs_ppl_error"] for r in rows], dtype=np.float64)
    rel_errors = np.asarray([r["rel_ppl_error"] for r in rows], dtype=np.float64)
    log_errors = np.asarray([r["log_ratio_error"] for r in rows], dtype=np.float64)
    mask_rows = np.asarray([r["drop_rate"] > 0 for r in rows])
    chosen_metrics = metric_block(y, pred_log, mask)
    metrics = {
        "ppl_ref": ppl_ref,
        "gamma": gamma,
        "points": len(rows),
        "fit": fit,
        "exponential_r2_log_ratio": exp_metrics["r2_log_ratio"],
        "exponential_rmse_log_ratio": exp_metrics["rmse_log_ratio"],
        "r2_log_ratio": chosen_metrics["r2_log_ratio"],
        "rmse_log_ratio": float(np.sqrt(np.mean(log_errors[mask_rows] ** 2))) if np.any(mask_rows) else 0.0,
        "mae_ppl": float(np.mean(abs_errors)),
        "max_abs_ppl_error": float(np.max(abs_errors)),
        "mean_relative_ppl_error": float(np.mean(rel_errors)),
        "max_relative_ppl_error": float(np.max(rel_errors)),
    }
    if coef is not None:
        metrics["poly_coefficients"] = [float(v) for v in coef]

    write_csv(out_dir / "surrogate_rows.csv", rows)
    (out_dir / "surrogate_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.write_surrogate:
        surrogate_payload = {
            "type": "scalar_curve_v1",
            "fit_model": fit,
            "source": (out_dir / "surrogate_metrics.json").as_posix(),
            "ppl_ref": ppl_ref,
            "x": [float(v) for v in drops],
            "y_log_ratio": [float(v) for v in pred_log],
        }
        (real_dir / "ppl_surrogate_curve.json").write_text(json.dumps(surrogate_payload, indent=2), encoding="utf-8")
    save_curve(rows, out_dir, fit_label)

    lines = [
        "# Surrogate Benchmark",
        "",
        f"Real profile directory: `{real_dir.as_posix()}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| PPL_ref | {metrics['ppl_ref']:.6f} |",
        f"| fit | {metrics['fit']} |",
        f"| gamma | {metrics['gamma']:.6f} |",
        f"| layer gamma sum | {float(layer_summary['fitted_gamma_sum']):.6f} |" if layer_summary else "| layer gamma sum | unavailable |",
        f"| layer mean R2 | {float(layer_summary['fitted_r2']):.6f} |" if layer_summary else "| layer mean R2 | unavailable |",
        f"| exponential R2 log-ratio | {metrics['exponential_r2_log_ratio']:.6f} |",
        f"| R2 log-ratio | {metrics['r2_log_ratio']:.6f} |",
        f"| RMSE log-ratio | {metrics['rmse_log_ratio']:.6f} |",
        f"| MAE PPL | {metrics['mae_ppl']:.6f} |",
        f"| Max abs PPL error | {metrics['max_abs_ppl_error']:.6f} |",
        f"| Mean relative PPL error | {metrics['mean_relative_ppl_error']:.6f} |",
        f"| Max relative PPL error | {metrics['max_relative_ppl_error']:.6f} |",
        "",
        "| drop_rate | real PPL | surrogate PPL | rel error |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['drop_rate']:.3f} | {row['real_ppl_mean']:.6f} +/- {row['real_ppl_std']:.6f} | "
            f"{row['surrogate_ppl']:.6f} | {row['rel_ppl_error']:.6f} |"
        )
    (out_dir / "surrogate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir / 'surrogate_report.md'}")


if __name__ == "__main__":
    main()
