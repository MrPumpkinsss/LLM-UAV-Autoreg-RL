from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_qwen35_surrogate(real_dir: Path) -> dict:
    mlp = load_json(real_dir / "ppl_surrogate_mlp.json")
    layer = load_json(real_dir / "layer_ppl_summary.json")
    return {
        "ppl_ref": float(mlp["ppl_ref"]),
        "mlp_r2": float(mlp["r2"]),
        "mlp_rmse_log_ratio": float(mlp["rmse_log_ratio"]),
        "mlp_mae_log_ratio": float(mlp["mae_log_ratio"]),
        "layer_gamma_sum": float(layer["fitted_gamma_sum"]),
        "layer_r2": float(layer["fitted_r2"]),
        "layer_rmse_log_ratio": float(layer["fitted_rmse_log_ratio"]),
        "rows": int(mlp["rows"]),
    }


def load_gemma_surrogate(real_dir: Path) -> dict:
    summary = load_json(real_dir / "real_profile_summary.json")
    curve = load_json(real_dir / "ppl_surrogate_curve.json")
    return {
        "ppl_ref": float(summary["ppl_ref"]),
        "fit_model": str(summary.get("surrogate_fit_model", curve.get("fit_model", "piecewise"))),
        "curve_points": int(summary.get("surrogate_curve_points", len(curve.get("x", [])))),
        "linear_r2": float(summary.get("surrogate_linear_r2", summary.get("surrogate_fit_r2", 0.0))),
        "linear_rmse_log_ratio": float(summary.get("surrogate_linear_rmse_log_ratio", summary.get("surrogate_fit_rmse_log_ratio", 0.0))),
        "fit_r2": float(summary.get("surrogate_fit_r2", 0.0)),
        "fit_rmse_log_ratio": float(summary.get("surrogate_fit_rmse_log_ratio", 0.0)),
    }


def fmt(v: float | int | str) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return f"{float(v):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3-real-dir", default="results/qwen3_0p6b_real_profile")
    parser.add_argument("--qwen35-real-dir", default="results/qwen35_4b_real_profile_v2")
    parser.add_argument("--gemma-real-dir", default="results/gemma4_e4b_real_profile")
    parser.add_argument("--qwen3-surrogate-dir", default="results/surrogate_benchmark_qwen3_0p6b")
    parser.add_argument("--qwen35-surrogate-dir", default="results/surrogate_benchmark_qwen35_4b_v2")
    parser.add_argument("--gemma-surrogate-dir", default="results/surrogate_benchmark_gemma4_e4b")
    parser.add_argument("--out", default="results/surrogate_benchmark_multi_model_report.md")
    args = parser.parse_args()

    qwen3_real = Path(args.qwen3_real_dir)
    qwen35_real = Path(args.qwen35_real_dir)
    gemma_real = Path(args.gemma_real_dir)
    qwen3_sur = Path(args.qwen3_surrogate_dir)
    qwen35_sur = Path(args.qwen35_surrogate_dir)
    gemma_sur = Path(args.gemma_surrogate_dir)

    qwen3_summary = load_json(qwen3_real / "real_profile_summary.json")
    qwen35_summary = load_json(qwen35_real / "real_profile_summary.json")
    gemma_summary = load_json(gemma_real / "real_profile_summary.json")

    qwen3_layer = load_json(qwen3_real / "layer_ppl_summary.json")
    qwen35_layer = load_json(qwen35_real / "layer_ppl_summary.json")

    qwen3_sur_metrics = load_json(qwen3_sur / "surrogate_metrics.json")
    qwen35_sur_metrics = load_qwen35_surrogate(qwen35_real)
    gemma_sur_metrics = load_json(gemma_sur / "surrogate_metrics.json")
    gemma_fit = load_gemma_surrogate(gemma_real)

    lines = [
        "# Multi-Model Surrogate Benchmark",
        "",
        "This report separates the three model families used in the repository. Qwen3-0.6B and Qwen3.5-4B use calibrated layer-wise surrogate fits; Gemma-4-E4B uses an empirical piecewise curve surrogate over the scalar damage proxy.",
        "",
        "| model | profile dir | clean PPL | surrogate R2 | RMSE log-ratio | notes |",
        "|---|---|---:|---:|---:|---|",
        f"| Qwen3-0.6B | `{qwen3_real.as_posix()}` | {fmt(qwen3_summary['ppl_ref'])} | {fmt(qwen3_sur_metrics['r2_log_ratio'])} | {fmt(qwen3_sur_metrics['rmse_log_ratio'])} | layer gamma sum {fmt(qwen3_layer['fitted_gamma_sum'])} |",
        f"| Qwen3.5-4B | `{qwen35_real.as_posix()}` | {fmt(qwen35_sur_metrics['ppl_ref'])} | {fmt(qwen35_sur_metrics['mlp_r2'])} | {fmt(qwen35_sur_metrics['mlp_rmse_log_ratio'])} | layer R2 {fmt(qwen35_sur_metrics['layer_r2'])}, rows {fmt(qwen35_sur_metrics['rows'])} |",
        f"| Gemma-4-E4B | `{gemma_real.as_posix()}` | {fmt(gemma_summary['ppl_ref'])} | {fmt(gemma_fit['fit_r2'])} | {fmt(gemma_fit['fit_rmse_log_ratio'])} | {gemma_fit['fit_model']} over {gemma_fit['curve_points']} points; linear R2 {fmt(gemma_fit['linear_r2'])} |",
        "",
        "### Qwen3-0.6B",
        "",
        f"- calibration: layer R2 `{fmt(qwen3_layer['fitted_r2'])}`",
        f"- surrogate benchmark: mean relative PPL error `{fmt(qwen3_sur_metrics['mean_relative_ppl_error'])}`",
        f"- max relative PPL error `{fmt(qwen3_sur_metrics['max_relative_ppl_error'])}`",
        "",
        "### Qwen3.5-4B",
        "",
        f"- calibration: layer R2 `{fmt(qwen35_sur_metrics['layer_r2'])}`",
        f"- layer RMSE log-ratio: `{fmt(qwen35_sur_metrics['layer_rmse_log_ratio'])}`",
        f"- MLP surrogate fit: R2 `{fmt(qwen35_sur_metrics['mlp_r2'])}`",
        f"- MLP surrogate RMSE log-ratio: `{fmt(qwen35_sur_metrics['mlp_rmse_log_ratio'])}`",
        "",
        "### Gemma-4-E4B",
        "",
        f"- curve fit model: `{gemma_fit['fit_model']}`",
        f"- curve fit R2: `{fmt(gemma_fit['fit_r2'])}`",
        f"- linear baseline R2: `{fmt(gemma_fit['linear_r2'])}`",
        f"- surrogate benchmark mean relative PPL error: `{fmt(gemma_sur_metrics['mean_relative_ppl_error'])}`",
        f"- surrogate benchmark max relative PPL error: `{fmt(gemma_sur_metrics['max_relative_ppl_error'])}`",
        "- interpretation: the curve surrogate is an empirical fit on sampled points; it is stronger than the old exponential baseline, but it is still not a layer-wise calibration.",
        "",
        "### Standalone surrogate benchmark directories",
        "",
        f"- Qwen3-0.6B: `{qwen3_sur.as_posix()}`",
        f"- Qwen3.5-4B: `{qwen35_sur.as_posix()}`",
        f"- Gemma-4-E4B: `{gemma_sur.as_posix()}`",
    ]

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
