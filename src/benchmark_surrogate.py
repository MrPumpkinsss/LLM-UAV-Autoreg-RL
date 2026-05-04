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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_curve(rows: list[dict], out_dir: Path) -> None:
    xs = np.asarray([r["drop_rate"] for r in rows], dtype=np.float64)
    y_real = np.asarray([r["real_ppl_mean"] for r in rows], dtype=np.float64)
    y_pred = np.asarray([r["surrogate_ppl"] for r in rows], dtype=np.float64)
    y_std = np.asarray([r["real_ppl_std"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(xs, y_real, yerr=y_std, fmt="o", color="#1f77b4", capsize=4, label="Real Qwen3-0.6B")
    ax.plot(xs, y_pred, color="#d62728", lw=2.0, label="Exponential surrogate")
    ax.set_xlabel("Embedding Drop Rate")
    ax.set_ylabel("Perplexity")
    ax.set_title("Surrogate PPL Fit")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "surrogate_ppl_fit.png", dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default="results/qwen3_0p6b_real_profile")
    parser.add_argument("--out", default="results/surrogate_benchmark")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    out_dir = ensure_dir(args.out)
    summary = json.loads((real_dir / "real_profile_summary.json").read_text(encoding="utf-8"))
    curve = json.loads((real_dir / "ppl_corruption_curve.json").read_text(encoding="utf-8"))
    layer_summary_path = real_dir / "layer_ppl_summary.json"
    layer_summary = json.loads(layer_summary_path.read_text(encoding="utf-8")) if layer_summary_path.exists() else None
    ppl_ref = float(summary["ppl_ref"])
    gamma = float(summary["fitted_gamma"])

    rows = []
    for item in curve:
        drop = float(item["drop_rate"])
        real_ppl = float(item["ppl_mean"])
        pred_ppl = float(ppl_ref * np.exp(gamma * drop))
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
    y = np.log(np.asarray([r["real_ppl_mean"] for r in rows], dtype=np.float64) / max(ppl_ref, 1e-12))
    pred = gamma * np.asarray([r["drop_rate"] for r in rows], dtype=np.float64)
    mask = np.asarray([r["drop_rate"] > 0 for r in rows])
    ss_res = float(np.sum((y[mask] - pred[mask]) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    metrics = {
        "ppl_ref": ppl_ref,
        "gamma": gamma,
        "points": len(rows),
        "r2_log_ratio": r2,
        "rmse_log_ratio": float(np.sqrt(np.mean(log_errors[mask] ** 2))) if np.any(mask) else 0.0,
        "mae_ppl": float(np.mean(abs_errors)),
        "max_abs_ppl_error": float(np.max(abs_errors)),
        "mean_relative_ppl_error": float(np.mean(rel_errors)),
        "max_relative_ppl_error": float(np.max(rel_errors)),
    }

    write_csv(out_dir / "surrogate_rows.csv", rows)
    (out_dir / "surrogate_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_curve(rows, out_dir)

    lines = [
        "# Surrogate Benchmark",
        "",
        f"Real profile directory: `{real_dir.as_posix()}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| PPL_ref | {metrics['ppl_ref']:.6f} |",
        f"| gamma | {metrics['gamma']:.6f} |",
        f"| layer gamma sum | {float(layer_summary['fitted_gamma_sum']):.6f} |" if layer_summary else "| layer gamma sum | unavailable |",
        f"| layer mean R2 | {float(layer_summary['fitted_r2']):.6f} |" if layer_summary else "| layer mean R2 | unavailable |",
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
