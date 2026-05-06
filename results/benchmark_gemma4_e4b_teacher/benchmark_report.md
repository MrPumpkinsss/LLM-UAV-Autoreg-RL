# Real google/gemma-4-E4B Profile Benchmark

States: `192`
Real profile directory: `results/gemma4_e4b_real_profile`

| method | reward | feasible | latency | PPL | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.277215 +/- 0.041744 | 1.0000 | 9.1506 | 10.8171 | 0.07353 |
| hybrid_heuristic | -0.277479 +/- 0.041227 | 1.0000 | 9.1666 | 10.8113 | 0.01737 |
| block_lns_strong | -0.277479 +/- 0.041227 | 1.0000 | 9.1666 | 10.8113 | 0.05510 |
| block_beam_strong | -0.280392 +/- 0.042654 | 1.0000 | 9.2573 | 10.8165 | 0.49193 |
| beam_search | -0.301616 +/- 0.059275 | 1.0000 | 9.9213 | 10.8515 | 0.09442 |
| simulated_annealing | -0.322986 +/- 0.077293 | 1.0000 | 10.4188 | 11.0246 | 0.00634 |
| local_search | -0.323179 +/- 0.077458 | 1.0000 | 10.4242 | 11.0254 | 0.02138 |
| pdp_aware_greedy | -0.336215 +/- 0.089650 | 1.0000 | 10.7860 | 11.0841 | 0.00084 |
| latency_greedy | -9.322157 +/- 28.337208 | 0.9115 | 73.9567 | 12.5073 | 0.00007 |
| block_balanced | -18.864876 +/- 38.409178 | 0.8177 | 82.1289 | 13.2769 | 0.00007 |
| random | -100.000000 +/- 0.000000 | 0.0000 | 577.5936 | 28.3381 | 0.03127 |

Autoreg-RL-pure margin vs best heuristic: mean `0.00026387`, min `-0.03373687`, win/tie rate `0.7396`.
