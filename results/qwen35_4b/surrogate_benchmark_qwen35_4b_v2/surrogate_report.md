# Qwen3.5-4B Surrogate Benchmark

Real profile directory: `results/qwen35_4b/qwen35_4b_real_profile_v2`

Qwen3.5-4B uses the tracked layer-wise calibration plus a layer-onehot MLP surrogate trained from the layer corruption curve. The embedding-level corruption curve is not present in this curated profile directory, so this report is intentionally layer/MLP based.

| metric | value |
|---|---:|
| PPL_ref | 12.388740 |
| layer gamma sum | 206.094048 |
| layer mean R2 | 0.995787 |
| layer RMSE log-ratio | 0.006596 |
| MLP R2 log-ratio | 0.999725 |
| MLP RMSE log-ratio | 0.001998 |
| MLP MAE log-ratio | 0.001143 |
| training rows | 155 |

Artifact: `results/qwen35_4b/qwen35_4b_real_profile_v2/ppl_surrogate_mlp.npz`
