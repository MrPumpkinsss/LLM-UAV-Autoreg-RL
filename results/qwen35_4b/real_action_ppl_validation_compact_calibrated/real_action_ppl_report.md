# Real Action-Level PPL Validation

Real profile directory: `results/qwen35_4b/qwen35_4b_real_profile_v2`
Policy: `results/qwen35_4b/autoreg_rl_qwen35_4b_v2_teacher_big/autoreg_policy_best.pt`
Action calibration: `results/qwen35_4b/qwen35_4b_real_profile_v2/action_ppl_calibration.json`
Rows: `32`

| scope | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| all | 32 | 0.005947 | 0.030636 | 0.008903 | 0.793284 | 0.672287 |
| non-random competitive | 32 | 0.005947 | 0.030636 | 0.008903 | 0.793284 | 0.672287 |

## Real LLM Method Benchmark

This table substitutes measured real LLM PPL into the same reward formula used by the simulator.

| method | rows | real reward | surrogate reward | latency | real PPL | surrogate PPL | mean rel error | transitions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block_lns_strong | 8 | -0.161081 | -0.162024 | 3.9751 | 13.6842 | 13.7134 | 0.005894 | 2.00 |
| hybrid_heuristic | 8 | -0.161832 | -0.162024 | 3.9751 | 13.7075 | 13.7134 | 0.004944 | 2.00 |
| block_beam_strong | 8 | -0.162025 | -0.162148 | 3.9788 | 13.7100 | 13.7138 | 0.003520 | 2.00 |
| autoreg_rl_pure | 8 | -0.168045 | -0.166715 | 4.0230 | 13.8554 | 13.8142 | 0.009429 | 2.00 |

`autoreg_rl_pure` real-reward margin vs best non-RL: mean `-0.007946`, min `-0.029679`, win/tie `0.2500`, strict win `0.1250` over `8` states.

| method | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure | 8 | 0.009429 | 0.030636 | 0.013573 | 0.858166 | 0.714286 |
| block_beam_strong | 8 | 0.003520 | 0.008829 | 0.004418 | 0.882728 | 0.690476 |
| block_lns_strong | 8 | 0.005894 | 0.016870 | 0.008433 | 0.538212 | 0.619048 |
| hybrid_heuristic | 8 | 0.004944 | 0.011178 | 0.006493 | 0.737130 | 0.666667 |

| method | surrogate PPL | real PPL | rel error | transitions |
|---|---:|---:|---:|---:|
| autoreg_rl_pure | 14.126306 | 14.289564 +/- 0.000000 | 0.011425 | 2 |
| block_beam_strong | 13.611390 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| block_lns_strong | 13.611390 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| hybrid_heuristic | 13.611390 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| autoreg_rl_pure | 14.100331 | 14.545961 +/- 0.000000 | 0.030636 | 2 |
| block_beam_strong | 13.937016 | 14.003216 +/- 0.000000 | 0.004728 | 2 |
| block_lns_strong | 13.937724 | 13.706497 +/- 0.000000 | 0.016870 | 2 |
| hybrid_heuristic | 13.937724 | 13.794624 +/- 0.000000 | 0.010374 | 2 |
| autoreg_rl_pure | 13.631385 | 13.638624 +/- 0.000000 | 0.000531 | 2 |
| block_beam_strong | 13.635337 | 13.638624 +/- 0.000000 | 0.000241 | 2 |
| block_lns_strong | 13.631471 | 13.638624 +/- 0.000000 | 0.000524 | 2 |
| hybrid_heuristic | 13.631471 | 13.638624 +/- 0.000000 | 0.000524 | 2 |
| autoreg_rl_pure | 13.611394 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| block_beam_strong | 13.611394 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| block_lns_strong | 13.611394 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| hybrid_heuristic | 13.611394 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| autoreg_rl_pure | 13.845167 | 13.638624 +/- 0.000000 | 0.015144 | 2 |
| block_beam_strong | 13.845167 | 13.724002 +/- 0.000000 | 0.008829 | 2 |
| block_lns_strong | 13.845167 | 13.935391 +/- 0.000000 | 0.006474 | 2 |
| hybrid_heuristic | 13.845167 | 13.692112 +/- 0.000000 | 0.011178 | 2 |
| autoreg_rl_pure | 13.611438 | 13.638624 +/- 0.000000 | 0.001993 | 2 |
| block_beam_strong | 13.611438 | 13.638624 +/- 0.000000 | 0.001993 | 2 |
| block_lns_strong | 13.611438 | 13.638624 +/- 0.000000 | 0.001993 | 2 |
| hybrid_heuristic | 13.611438 | 13.638624 +/- 0.000000 | 0.001993 | 2 |
| autoreg_rl_pure | 13.976486 | 13.814763 +/- 0.000000 | 0.011707 | 2 |
| block_beam_strong | 13.847343 | 13.759605 +/- 0.000000 | 0.006377 | 2 |
| block_lns_strong | 13.847343 | 13.638624 +/- 0.000000 | 0.015304 | 2 |
| hybrid_heuristic | 13.847343 | 13.980063 +/- 0.000000 | 0.009494 | 2 |
| autoreg_rl_pure | 13.611384 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| block_beam_strong | 13.611384 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| block_lns_strong | 13.611384 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
| hybrid_heuristic | 13.611384 | 13.638624 +/- 0.000000 | 0.001997 | 2 |
