# Real Qwen3-0.6B Profile Benchmark

States: `320`
Real profile directory: `results/qwen3_0p6b_real_profile`

| method | reward | feasible | latency | PPL | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.256232 +/- 0.066645 | 1.0000 | 2.4984 | 31.3171 | 0.02587 |
| hybrid_heuristic | -0.273453 +/- 0.086901 | 1.0000 | 2.6411 | 31.5446 | 0.01061 |
| beam_search | -0.278149 +/- 0.094707 | 1.0000 | 2.6786 | 31.6178 | 0.01061 |
| simulated_annealing | -0.351507 +/- 0.146873 | 1.0000 | 3.2171 | 33.1205 | 0.01061 |
| local_search | -0.352602 +/- 0.148446 | 1.0000 | 3.2242 | 33.1506 | 0.01061 |
| pdp_aware_greedy | -0.369239 +/- 0.178122 | 1.0000 | 3.3570 | 33.4093 | 0.01061 |
| latency_greedy | -6.340535 +/- 22.905382 | 0.9437 | 8.3329 | 43.5884 | 0.01061 |
| block_balanced | -13.781014 +/- 33.106199 | 0.8719 | 12.7195 | 48.6002 | 0.01061 |
| random | -100.000000 +/- 0.000000 | 0.0000 | 109.9627 | 102.5448 | 0.01061 |

Autoreg-RL-pure margin vs best heuristic: mean `0.01722135`, min `-0.02921769`, win/tie rate `0.9156`.
