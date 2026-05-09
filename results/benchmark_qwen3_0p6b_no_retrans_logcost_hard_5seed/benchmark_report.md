# Real Qwen/Qwen3-0.6B Profile Benchmark

States: `320`
Real profile directory: `results/qwen3_0p6b_real_profile_raw_dense`

| method | reward | feasible | latency | PPL | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.257262 +/- 0.059339 | 1.0000 | 2.3714 | 32.4630 | 0.11442 |
| hybrid_heuristic | -0.259101 +/- 0.061291 | 1.0000 | 2.3828 | 32.5285 | 0.01378 |
| block_lns_strong | -0.259101 +/- 0.061291 | 1.0000 | 2.3828 | 32.5285 | 0.06987 |
| block_beam_strong | -0.264854 +/- 0.064152 | 1.0000 | 2.4271 | 32.6427 | 0.43397 |
| beam_search | -0.291034 +/- 0.092573 | 1.0000 | 2.6276 | 33.1968 | 0.08549 |
| simulated_annealing | -0.347934 +/- 0.125584 | 1.0000 | 2.8706 | 36.2588 | 0.00723 |
| local_search | -0.348585 +/- 0.126050 | 1.0000 | 2.8752 | 36.2784 | 0.01641 |
| pdp_aware_greedy | -0.366977 +/- 0.147391 | 1.0000 | 3.0185 | 36.7050 | 0.00079 |
| latency_greedy | -4.078920 +/- 18.129531 | 0.9656 | 5.3412 | 50.9341 | 0.00008 |
| block_balanced | -6.529624 +/- 22.858047 | 0.9437 | 8.1595 | 58.8727 | 0.00008 |
| random | -100.000000 +/- 0.000000 | 0.0000 | 80.9166 | 89.7442 | 0.02467 |

Autoreg-RL-pure margin vs best heuristic: mean `0.00183877`, min `-0.02780817`, win/tie rate `0.8938`.
