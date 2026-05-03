# Real Qwen3-0.6B Profile Benchmark

States: `192`
Real profile directory: `results/qwen3_0p6b_real_profile`

| method | reward | feasible | latency | PPL | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.261002 +/- 0.072367 | 1.0000 | 2.5350 | 31.4028 | 0.01985 |
| hybrid_heuristic | -0.269523 +/- 0.082058 | 1.0000 | 2.6010 | 31.5508 | 0.01045 |
| beam_search | -0.273802 +/- 0.087047 | 1.0000 | 2.6352 | 31.6174 | 0.01045 |
| simulated_annealing | -0.344778 +/- 0.133848 | 1.0000 | 3.1465 | 33.1463 | 0.01045 |
| local_search | -0.345173 +/- 0.133750 | 1.0000 | 3.1503 | 33.1472 | 0.01045 |
| pdp_aware_greedy | -0.357012 +/- 0.144814 | 1.0000 | 3.2462 | 33.3209 | 0.01045 |
| latency_greedy | -4.895185 +/- 19.886923 | 0.9583 | 8.3558 | 43.4784 | 0.01045 |
| block_balanced | -10.417073 +/- 28.891749 | 0.9062 | 12.6621 | 48.6316 | 0.01045 |
| random | -100.000000 +/- 0.000000 | 0.0000 | 110.8475 | 108.7251 | 0.01045 |

Autoreg-RL-pure margin vs best heuristic: mean `0.00852120`, min `-0.12508317`, win/tie rate `0.6406`.
