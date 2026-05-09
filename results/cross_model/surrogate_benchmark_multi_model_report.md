# Multi-Model Surrogate Benchmark

This report separates the three model families used in the repository. Qwen3-0.6B and Qwen3.5-4B use calibrated layer-wise surrogate fits; Gemma-4-E4B uses an empirical piecewise curve surrogate over the scalar damage proxy.

| model | profile dir | clean PPL | surrogate R2 | RMSE log-ratio | notes |
|---|---|---:|---:|---:|---|
| Qwen3-0.6B | `results/qwen3_0p6b/qwen3_0p6b_real_profile` | 30.811979 | 0.997363 | 0.019289 | layer gamma sum 293.322997 |
| Qwen3.5-4B | `results/qwen35_4b/qwen35_4b_real_profile_v2` | 12.388740 | 0.999725 | 0.001998 | layer R2 0.995787, rows 155 |
| Gemma-4-E4B | `results/gemma4_e4b/gemma4_e4b_real_profile` | 10.744652 | 1.000000 | 0.000000 | piecewise over 9 points; linear R2 0.891476 |

### Qwen3-0.6B

- calibration: layer R2 `0.982527`
- surrogate benchmark: mean relative PPL error `0.015285`
- max relative PPL error `0.030760`

### Qwen3.5-4B

- calibration: layer R2 `0.995787`
- layer RMSE log-ratio: `0.006596`
- MLP surrogate fit: R2 `0.999725`
- MLP surrogate RMSE log-ratio: `0.001998`

### Gemma-4-E4B

- curve fit model: `piecewise`
- curve fit R2: `1.000000`
- linear baseline R2: `0.891476`
- surrogate benchmark mean relative PPL error: `0.000000`
- surrogate benchmark max relative PPL error: `0.000000`
- interpretation: the curve surrogate is an empirical fit on sampled points; it is stronger than the old exponential baseline, but it is still not a layer-wise calibration.

### Standalone surrogate benchmark directories

- Qwen3-0.6B: `results/qwen3_0p6b/surrogate_benchmark_qwen3_0p6b`
- Qwen3.5-4B: `results/qwen35_4b/surrogate_benchmark_qwen35_4b_v2`
- Gemma-4-E4B: `results/gemma4_e4b/surrogate_benchmark_gemma4_e4b`
