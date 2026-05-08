# Real google/gemma-4-E4B Profile Benchmark

States: `192`
Real profile directory: `results/gemma4_e4b_real_profile`

| method | reward | feasible | latency | PPL | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.278327 +/- 0.042406 | 1.0000 | 9.2074 | 10.8012 | 0.07182 |
| hybrid_heuristic | -0.278862 +/- 0.041358 | 1.0000 | 9.2292 | 10.7980 | 0.01851 |
| block_lns_strong | -0.278862 +/- 0.041358 | 1.0000 | 9.2292 | 10.7980 | 0.06608 |
| block_beam_strong | -0.281125 +/- 0.042539 | 1.0000 | 9.3036 | 10.7988 | 0.68703 |
| beam_search | -0.304512 +/- 0.064632 | 1.0000 | 10.0208 | 10.8491 | 0.14239 |
| simulated_annealing | -0.329265 +/- 0.088857 | 1.0000 | 10.4656 | 11.1555 | 0.00743 |
| local_search | -0.329393 +/- 0.089003 | 1.0000 | 10.4692 | 11.1561 | 0.02313 |
| pdp_aware_greedy | -0.343542 +/- 0.101166 | 1.0000 | 10.7955 | 11.2732 | 0.00127 |
| latency_greedy | -10.915209 +/- 30.458297 | 0.8958 | 28.5098 | 13.6214 | 0.00018 |
| block_balanced | -22.505554 +/- 41.114116 | 0.7812 | 38.5714 | 14.5793 | 0.00009 |
| random | -100.000000 +/- 0.000000 | 0.0000 | 569.4376 | 26.9634 | 0.02984 |

Autoreg-RL-pure margin vs best heuristic: mean `0.00053444`, min `-0.04038793`, win/tie rate `0.7448`.
