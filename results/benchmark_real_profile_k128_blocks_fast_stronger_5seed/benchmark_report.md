# Real Qwen3-0.6B Profile Benchmark

States: `320`
Real profile directory: `results/qwen3_0p6b_real_profile`

| method | reward | feasible | latency | PPL | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.250877 +/- 0.064429 | 1.0000 | 2.4492 | 31.2834 | 0.02585 |
| hybrid_heuristic | -0.251173 +/- 0.065266 | 1.0000 | 2.4497 | 31.3028 | 0.01191 |
| block_lns_strong | -0.251173 +/- 0.065266 | 1.0000 | 2.4497 | 31.3028 | 0.06159 |
| block_beam_strong | -0.257286 +/- 0.068740 | 1.0000 | 2.5078 | 31.3262 | 0.30722 |
| beam_search | -0.270223 +/- 0.088483 | 1.0000 | 2.6111 | 31.5269 | 0.02989 |
| simulated_annealing | -0.353110 +/- 0.153723 | 1.0000 | 3.2136 | 33.2715 | 0.00639 |
| local_search | -0.353971 +/- 0.155190 | 1.0000 | 3.2201 | 33.2875 | 0.01430 |
| pdp_aware_greedy | -0.686098 +/- 5.572170 | 0.9969 | 3.4235 | 33.6823 | 0.00033 |
| latency_greedy | -7.903879 +/- 25.672935 | 0.9281 | 7.8935 | 43.4328 | 0.00007 |
| block_balanced | -18.696634 +/- 37.911727 | 0.8219 | 12.4099 | 48.5239 | 0.00008 |
| random | -100.000000 +/- 0.000000 | 0.0000 | 110.3007 | 104.9469 | 0.02294 |

Autoreg-RL-pure margin vs best heuristic: mean `0.00029552`, min `-0.07380839`, win/tie rate `0.8156`.
