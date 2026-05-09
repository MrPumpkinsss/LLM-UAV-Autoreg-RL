# Real Action-Level PPL Validation

Real profile directory: `results/qwen35_4b/qwen35_4b_real_profile_v2`
Policy: `results/qwen35_4b/autoreg_rl_qwen35_4b_v2_teacher_big/autoreg_policy_best.pt`
Rows: `32`

| scope | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| all | 32 | 0.093873 | 0.132779 | 0.099194 | 0.792281 | 0.672287 |
| non-random competitive | 32 | 0.093873 | 0.132779 | 0.099194 | 0.792281 | 0.672287 |

| method | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure | 8 | 0.098717 | 0.132779 | 0.105567 | 0.857333 | 0.714286 |
| block_beam_strong | 8 | 0.092873 | 0.104522 | 0.097610 | 0.882252 | 0.690476 |
| block_lns_strong | 8 | 0.091183 | 0.103204 | 0.095819 | 0.538575 | 0.619048 |
| hybrid_heuristic | 8 | 0.092719 | 0.105998 | 0.097491 | 0.737374 | 0.666667 |

| method | surrogate PPL | real PPL | rel error | transitions |
|---|---:|---:|---:|---:|
| autoreg_rl_pure | 12.626459 | 14.289564 +/- 0.000000 | 0.116386 | 2 |
| block_beam_strong | 12.388743 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| block_lns_strong | 12.388743 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| hybrid_heuristic | 12.388743 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| autoreg_rl_pure | 12.614570 | 14.545961 +/- 0.000000 | 0.132779 | 2 |
| block_beam_strong | 12.539571 | 14.003216 +/- 0.000000 | 0.104522 | 2 |
| block_lns_strong | 12.539897 | 13.706497 +/- 0.000000 | 0.085113 | 2 |
| hybrid_heuristic | 12.539897 | 13.794624 +/- 0.000000 | 0.090958 | 2 |
| autoreg_rl_pure | 12.398054 | 13.638624 +/- 0.000000 | 0.090960 | 2 |
| block_beam_strong | 12.399894 | 13.638624 +/- 0.000000 | 0.090825 | 2 |
| block_lns_strong | 12.398095 | 13.638624 +/- 0.000000 | 0.090957 | 2 |
| hybrid_heuristic | 12.398095 | 13.638624 +/- 0.000000 | 0.090957 | 2 |
| autoreg_rl_pure | 12.388744 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| block_beam_strong | 12.388744 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| block_lns_strong | 12.388744 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| hybrid_heuristic | 12.388744 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| autoreg_rl_pure | 12.497202 | 13.638624 +/- 0.000000 | 0.083690 | 2 |
| block_beam_strong | 12.497202 | 13.724002 +/- 0.000000 | 0.089391 | 2 |
| block_lns_strong | 12.497202 | 13.935391 +/- 0.000000 | 0.103204 | 2 |
| hybrid_heuristic | 12.497202 | 13.692112 +/- 0.000000 | 0.087270 | 2 |
| autoreg_rl_pure | 12.388765 | 13.638624 +/- 0.000000 | 0.091641 | 2 |
| block_beam_strong | 12.388765 | 13.638624 +/- 0.000000 | 0.091641 | 2 |
| block_lns_strong | 12.388765 | 13.638624 +/- 0.000000 | 0.091641 | 2 |
| hybrid_heuristic | 12.388765 | 13.638624 +/- 0.000000 | 0.091641 | 2 |
| autoreg_rl_pure | 12.557735 | 13.814763 +/- 0.000000 | 0.090992 | 2 |
| block_beam_strong | 12.498208 | 13.759605 +/- 0.000000 | 0.091674 | 2 |
| block_lns_strong | 12.498208 | 13.638624 +/- 0.000000 | 0.083617 | 2 |
| hybrid_heuristic | 12.498208 | 13.980063 +/- 0.000000 | 0.105998 | 2 |
| autoreg_rl_pure | 12.388740 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| block_beam_strong | 12.388740 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| block_lns_strong | 12.388740 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
| hybrid_heuristic | 12.388740 | 13.638624 +/- 0.000000 | 0.091643 | 2 |
