# Real Action-Level PPL Validation

Real profile directory: `results/gemma4_e4b/gemma4_e4b_real_profile`
Policy: `results/gemma4_e4b/autoreg_rl_gemma4_e4b_teacher/autoreg_policy_best.pt`
Rows: `32`

| scope | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| all | 32 | 0.270824 | 0.620170 | 0.362919 | 0.921446 | 0.872067 |
| non-random competitive | 32 | 0.270824 | 0.620170 | 0.362919 | 0.921446 | 0.872067 |

| method | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure | 8 | 0.262776 | 0.461419 | 0.334477 | 0.989013 | 0.833333 |
| block_beam_strong | 8 | 0.310920 | 0.620170 | 0.465956 | 0.971316 | 0.976190 |
| block_lns_strong | 8 | 0.247540 | 0.381384 | 0.297032 | 0.990229 | 0.833333 |
| hybrid_heuristic | 8 | 0.262059 | 0.456648 | 0.331095 | 0.973646 | 0.857143 |

| method | surrogate PPL | real PPL | rel error | transitions |
|---|---:|---:|---:|---:|
| autoreg_rl_pure | 10.745224 | 13.699887 +/- 0.000000 | 0.215671 | 2 |
| block_beam_strong | 11.754662 | 30.947146 +/- 0.000000 | 0.620170 | 2 |
| block_lns_strong | 10.745224 | 13.699887 +/- 0.000000 | 0.215671 | 2 |
| hybrid_heuristic | 10.745224 | 13.699887 +/- 0.000000 | 0.215671 | 2 |
| autoreg_rl_pure | 10.744652 | 13.699887 +/- 0.000000 | 0.215712 | 2 |
| block_beam_strong | 10.744652 | 13.699887 +/- 0.000000 | 0.215712 | 2 |
| block_lns_strong | 10.744652 | 13.699887 +/- 0.000000 | 0.215712 | 2 |
| hybrid_heuristic | 10.744652 | 13.699887 +/- 0.000000 | 0.215712 | 2 |
| autoreg_rl_pure | 10.844701 | 20.135705 +/- 0.000000 | 0.461419 | 3 |
| block_beam_strong | 10.844701 | 19.376805 +/- 0.000000 | 0.440326 | 3 |
| block_lns_strong | 10.844701 | 17.530590 +/- 0.000000 | 0.381384 | 3 |
| hybrid_heuristic | 10.844701 | 19.958903 +/- 0.000000 | 0.456648 | 3 |
| autoreg_rl_pure | 10.744918 | 13.699887 +/- 0.000000 | 0.215693 | 2 |
| block_beam_strong | 10.744918 | 13.699887 +/- 0.000000 | 0.215693 | 2 |
| block_lns_strong | 10.744918 | 13.699887 +/- 0.000000 | 0.215693 | 2 |
| hybrid_heuristic | 10.744918 | 13.804335 +/- 0.000000 | 0.221627 | 2 |
| autoreg_rl_pure | 10.758403 | 14.013452 +/- 0.000000 | 0.232280 | 2 |
| block_beam_strong | 10.758403 | 13.886705 +/- 0.000000 | 0.225273 | 2 |
| block_lns_strong | 10.758403 | 13.985891 +/- 0.000000 | 0.230767 | 2 |
| hybrid_heuristic | 10.758403 | 14.402402 +/- 0.000000 | 0.253013 | 2 |
| autoreg_rl_pure | 10.781480 | 15.346120 +/- 0.000000 | 0.297446 | 2 |
| block_beam_strong | 10.781480 | 15.886123 +/- 0.000000 | 0.321327 | 2 |
| block_lns_strong | 10.782157 | 14.637355 +/- 0.000000 | 0.263381 | 2 |
| hybrid_heuristic | 10.782157 | 14.665467 +/- 0.000000 | 0.264793 | 2 |
| autoreg_rl_pure | 10.744655 | 13.699887 +/- 0.000000 | 0.215712 | 3 |
| block_beam_strong | 10.744655 | 13.699887 +/- 0.000000 | 0.215712 | 3 |
| block_lns_strong | 10.744655 | 13.699887 +/- 0.000000 | 0.215712 | 3 |
| hybrid_heuristic | 10.744655 | 13.699887 +/- 0.000000 | 0.215712 | 3 |
| autoreg_rl_pure | 10.764015 | 14.319021 +/- 0.000000 | 0.248272 | 2 |
| block_beam_strong | 10.764015 | 14.036614 +/- 0.000000 | 0.233147 | 2 |
| block_lns_strong | 10.764015 | 14.200520 +/- 0.000000 | 0.241999 | 2 |
| hybrid_heuristic | 10.764015 | 14.415325 +/- 0.000000 | 0.253294 | 2 |
