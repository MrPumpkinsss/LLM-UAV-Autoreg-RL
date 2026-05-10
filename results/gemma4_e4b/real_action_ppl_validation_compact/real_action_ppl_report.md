# Real Action-Level PPL Validation

Real profile directory: `results/gemma4_e4b/gemma4_e4b_real_profile`
Policy: `results/gemma4_e4b/autoreg_rl_gemma4_e4b_teacher/autoreg_policy_best.pt`
Action calibration: none
Autoregressive RL candidates: `256`
Rows: `32`

| scope | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| all | 32 | 0.264362 | 0.620170 | 0.349914 | 0.942891 | 0.859238 |
| non-random competitive | 32 | 0.264362 | 0.620170 | 0.349914 | 0.942891 | 0.859238 |

## Real LLM Method Benchmark

This table substitutes measured real LLM PPL into the same reward formula used by the simulator.

| method | rows | real reward | surrogate reward | latency | real PPL | surrogate PPL | mean rel error | transitions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure | 8 | -0.370592 | -0.245565 | 8.1731 | 14.1130 | 10.7546 | 0.236927 | 2.25 |
| block_lns_strong | 8 | -0.385024 | -0.249956 | 8.3053 | 14.3942 | 10.7661 | 0.247540 | 2.25 |
| hybrid_heuristic | 8 | -0.399878 | -0.249956 | 8.3053 | 14.7933 | 10.7661 | 0.262059 | 2.25 |
| block_beam_strong | 8 | -0.482219 | -0.258407 | 8.4305 | 16.9041 | 10.8922 | 0.310920 | 2.25 |

`autoreg_rl_pure` real-reward margin vs best non-RL: mean `0.013207`, min `-0.026096`, win/tie `0.6250`, strict win `0.2500` over `8` states.

| method | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure | 8 | 0.236927 | 0.297045 | 0.273482 | 0.942540 | 0.761905 |
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
| autoreg_rl_pure | 10.752883 | 14.434048 +/- 0.000000 | 0.255033 | 3 |
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
| autoreg_rl_pure | 10.782157 | 15.338323 +/- 0.000000 | 0.297045 | 2 |
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
