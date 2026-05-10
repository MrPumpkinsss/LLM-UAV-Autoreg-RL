# Real Action-Level PPL Validation

Real profile directory: `results/qwen35_4b/qwen35_4b_real_profile_v2`
Policy: `results/qwen35_4b/autoreg_rl_qwen35_4b_v2_teacher_big/autoreg_policy_best.pt`
Action calibration: `results/qwen35_4b/qwen35_4b_real_profile_v2/action_ppl_calibration.json`
Autoregressive RL candidates: `256`
Rows: `32`

| scope | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| all | 32 | 0.002744 | 0.015532 | 0.005241 | 0.706439 | 0.630865 |
| non-random competitive | 32 | 0.002744 | 0.015532 | 0.005241 | 0.706439 | 0.630865 |

## Real LLM Method Benchmark

This table substitutes measured real LLM PPL into the same reward formula used by the simulator.

| method | rows | real reward | surrogate reward | latency | real PPL | surrogate PPL | mean rel error | transitions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block_lns_strong | 8 | -0.161081 | -0.161313 | 3.9751 | 13.6842 | 13.6914 | 0.003793 | 2.00 |
| autoreg_rl_pure | 8 | -0.161440 | -0.162296 | 3.9987 | 13.6734 | 13.6999 | 0.002057 | 2.00 |
| hybrid_heuristic | 8 | -0.161832 | -0.161313 | 3.9751 | 13.7075 | 13.6914 | 0.002869 | 2.00 |
| block_beam_strong | 8 | -0.162025 | -0.161432 | 3.9788 | 13.7100 | 13.6916 | 0.002257 | 2.00 |

`autoreg_rl_pure` real-reward margin vs best non-RL: mean `-0.001341`, min `-0.006735`, win/tie `0.6250`, strict win `0.2500` over `8` states.

| method | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure | 8 | 0.002057 | 0.009029 | 0.003680 | 0.848463 | 0.595238 |
| block_beam_strong | 8 | 0.002257 | 0.013760 | 0.005004 | 0.882274 | 0.690476 |
| block_lns_strong | 8 | 0.003793 | 0.012459 | 0.006099 | 0.538557 | 0.619048 |
| hybrid_heuristic | 8 | 0.002869 | 0.015532 | 0.005838 | 0.737362 | 0.666667 |

| method | surrogate PPL | real PPL | rel error | transitions |
|---|---:|---:|---:|---:|
| autoreg_rl_pure | 13.636966 | 13.638624 +/- 0.000000 | 0.000122 | 2 |
| block_beam_strong | 13.636966 | 13.638624 +/- 0.000000 | 0.000122 | 2 |
| block_lns_strong | 13.636966 | 13.638624 +/- 0.000000 | 0.000122 | 2 |
| hybrid_heuristic | 13.636966 | 13.638624 +/- 0.000000 | 0.000122 | 2 |
| autoreg_rl_pure | 13.810539 | 13.740852 +/- 0.000000 | 0.005071 | 2 |
| block_beam_strong | 13.810539 | 14.003216 +/- 0.000000 | 0.013760 | 2 |
| block_lns_strong | 13.810914 | 13.706497 +/- 0.000000 | 0.007618 | 2 |
| hybrid_heuristic | 13.810914 | 13.794624 +/- 0.000000 | 0.001181 | 2 |
| autoreg_rl_pure | 13.647726 | 13.638624 +/- 0.000000 | 0.000667 | 2 |
| block_beam_strong | 13.649796 | 13.638624 +/- 0.000000 | 0.000819 | 2 |
| block_lns_strong | 13.647726 | 13.638624 +/- 0.000000 | 0.000667 | 2 |
| hybrid_heuristic | 13.647726 | 13.638624 +/- 0.000000 | 0.000667 | 2 |
| autoreg_rl_pure | 13.636982 | 13.638624 +/- 0.000000 | 0.000120 | 2 |
| block_beam_strong | 13.636968 | 13.638624 +/- 0.000000 | 0.000121 | 2 |
| block_lns_strong | 13.636968 | 13.638624 +/- 0.000000 | 0.000121 | 2 |
| hybrid_heuristic | 13.636968 | 13.638624 +/- 0.000000 | 0.000121 | 2 |
| autoreg_rl_pure | 13.761772 | 13.638624 +/- 0.000000 | 0.009029 | 2 |
| block_beam_strong | 13.761772 | 13.724002 +/- 0.000000 | 0.002752 | 2 |
| block_lns_strong | 13.761772 | 13.935391 +/- 0.000000 | 0.012459 | 2 |
| hybrid_heuristic | 13.761772 | 13.692112 +/- 0.000000 | 0.005088 | 2 |
| autoreg_rl_pure | 13.636992 | 13.638624 +/- 0.000000 | 0.000120 | 2 |
| block_beam_strong | 13.636992 | 13.638624 +/- 0.000000 | 0.000120 | 2 |
| block_lns_strong | 13.636992 | 13.638624 +/- 0.000000 | 0.000120 | 2 |
| hybrid_heuristic | 13.636992 | 13.638624 +/- 0.000000 | 0.000120 | 2 |
| autoreg_rl_pure | 13.831450 | 13.814763 +/- 0.000000 | 0.001208 | 2 |
| block_beam_strong | 13.762929 | 13.759605 +/- 0.000000 | 0.000242 | 2 |
| block_lns_strong | 13.762929 | 13.638624 +/- 0.000000 | 0.009114 | 2 |
| hybrid_heuristic | 13.762929 | 13.980063 +/- 0.000000 | 0.015532 | 2 |
| autoreg_rl_pure | 13.636963 | 13.638624 +/- 0.000000 | 0.000122 | 2 |
| block_beam_strong | 13.636963 | 13.638624 +/- 0.000000 | 0.000122 | 2 |
| block_lns_strong | 13.636963 | 13.638624 +/- 0.000000 | 0.000122 | 2 |
| hybrid_heuristic | 13.636963 | 13.638624 +/- 0.000000 | 0.000122 | 2 |
