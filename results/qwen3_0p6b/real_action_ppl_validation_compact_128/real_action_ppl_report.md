# Real Action-Level PPL Validation

Real profile directory: `results/qwen3_0p6b/qwen3_0p6b_real_profile`
Policy: `results/qwen3_0p6b/autoreg_rl_layer_calibrated_hard_k256/autoreg_policy_best.pt`
Action calibration: none
Rows: `32`

| scope | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| all | 32 | 0.028290 | 0.126255 | 0.043024 | 0.990272 | 0.985337 |
| non-random competitive | 32 | 0.028290 | 0.126255 | 0.043024 | 0.990272 | 0.985337 |

## Real LLM Method Benchmark

This table substitutes measured real LLM PPL into the same reward formula used by the simulator.

| method | rows | real reward | surrogate reward | latency | real PPL | surrogate PPL | mean rel error | transitions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_heuristic | 8 | -0.345564 | -0.366192 | 2.3536 | 39.3013 | 40.8903 | 0.037794 | 2.75 |
| block_lns_strong | 8 | -0.349865 | -0.366192 | 2.3536 | 39.6325 | 40.8903 | 0.024467 | 2.75 |
| block_beam_strong | 8 | -0.372417 | -0.367709 | 2.3630 | 41.2968 | 40.9342 | 0.015877 | 2.75 |
| autoreg_rl_pure | 8 | -0.389492 | -0.401658 | 2.4405 | 42.0151 | 42.9523 | 0.035020 | 2.88 |

`autoreg_rl_pure` real-reward margin vs best non-RL: mean `-0.048524`, min `-0.161586`, win/tie `0.3750`, strict win `0.1250` over `8` states.

| method | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure | 8 | 0.035020 | 0.126255 | 0.051552 | 0.991931 | 1.000000 |
| block_beam_strong | 8 | 0.015877 | 0.043138 | 0.023174 | 0.997505 | 1.000000 |
| block_lns_strong | 8 | 0.024467 | 0.082609 | 0.038192 | 0.999233 | 1.000000 |
| hybrid_heuristic | 8 | 0.037794 | 0.122371 | 0.052450 | 0.994508 | 1.000000 |

| method | surrogate PPL | real PPL | rel error | transitions |
|---|---:|---:|---:|---:|
| autoreg_rl_pure | 79.800569 | 75.199398 +/- 0.000000 | 0.061186 | 2 |
| block_beam_strong | 63.339110 | 65.390275 +/- 0.000000 | 0.031368 | 3 |
| block_lns_strong | 63.339110 | 59.164448 +/- 0.000000 | 0.070560 | 3 |
| hybrid_heuristic | 63.339110 | 60.205256 +/- 0.000000 | 0.052053 | 3 |
| autoreg_rl_pure | 34.297760 | 35.024992 +/- 0.000000 | 0.020763 | 3 |
| block_beam_strong | 34.610619 | 34.607240 +/- 0.000000 | 0.000098 | 3 |
| block_lns_strong | 34.297760 | 34.399343 +/- 0.000000 | 0.002953 | 3 |
| hybrid_heuristic | 34.297760 | 32.786358 +/- 0.000000 | 0.046099 | 3 |
| autoreg_rl_pure | 30.811979 | 30.843947 +/- 0.000000 | 0.001036 | 2 |
| block_beam_strong | 30.811979 | 30.843947 +/- 0.000000 | 0.001036 | 2 |
| block_lns_strong | 30.811979 | 30.843947 +/- 0.000000 | 0.001036 | 2 |
| hybrid_heuristic | 30.811979 | 30.843947 +/- 0.000000 | 0.001036 | 2 |
| autoreg_rl_pure | 30.945055 | 30.946976 +/- 0.000000 | 0.000062 | 4 |
| block_beam_strong | 30.951411 | 30.943845 +/- 0.000000 | 0.000245 | 3 |
| block_lns_strong | 30.951411 | 30.908830 +/- 0.000000 | 0.001378 | 3 |
| hybrid_heuristic | 30.951411 | 30.896544 +/- 0.000000 | 0.001776 | 3 |
| autoreg_rl_pure | 59.873614 | 53.161667 +/- 0.000000 | 0.126255 | 3 |
| block_beam_strong | 59.911914 | 62.073719 +/- 0.000000 | 0.034826 | 3 |
| block_lns_strong | 59.873614 | 55.304947 +/- 0.000000 | 0.082609 | 3 |
| hybrid_heuristic | 59.873614 | 56.259099 +/- 0.000000 | 0.064248 | 3 |
| autoreg_rl_pure | 44.363845 | 46.782712 +/- 0.000000 | 0.051704 | 3 |
| block_beam_strong | 45.008761 | 43.147469 +/- 0.000000 | 0.043138 | 2 |
| block_lns_strong | 45.008761 | 43.959005 +/- 0.000000 | 0.023880 | 2 |
| hybrid_heuristic | 45.008761 | 40.101505 +/- 0.000000 | 0.122371 | 2 |
| autoreg_rl_pure | 32.711918 | 33.317149 +/- 0.000000 | 0.018166 | 3 |
| block_beam_strong | 32.025989 | 32.524252 +/- 0.000000 | 0.015320 | 3 |
| block_lns_strong | 32.025989 | 31.635896 +/- 0.000000 | 0.012331 | 3 |
| hybrid_heuristic | 32.025989 | 32.473563 +/- 0.000000 | 0.013783 | 3 |
| autoreg_rl_pure | 30.813438 | 30.843947 +/- 0.000000 | 0.000989 | 3 |
| block_beam_strong | 30.813438 | 30.843947 +/- 0.000000 | 0.000989 | 3 |
| block_lns_strong | 30.813438 | 30.843947 +/- 0.000000 | 0.000989 | 3 |
| hybrid_heuristic | 30.813438 | 30.843947 +/- 0.000000 | 0.000989 | 3 |
