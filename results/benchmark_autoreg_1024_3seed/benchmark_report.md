# Strong Heuristic Benchmark

States: `192`

| method | reward | feasible | latency | PPL | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.197280 +/- 0.058722 | 1.0000 | 1.9311 | 31.1460 | 0.50797 |
| hybrid_heuristic | -0.198966 +/- 0.057822 | 1.0000 | 1.9493 | 31.1356 | 0.01748 |
| dros_hybrid | -0.198966 +/- 0.057822 | 1.0000 | 1.9493 | 31.1356 | 1.54695 |
| beam_search | -0.201209 +/- 0.062074 | 1.0000 | 1.9696 | 31.1523 | 0.01748 |
| simulated_annealing | -0.281337 +/- 0.128424 | 1.0000 | 2.5978 | 32.4855 | 0.01748 |
| local_search | -0.282460 +/- 0.129466 | 1.0000 | 2.6002 | 32.5542 | 0.01748 |
| pdp_aware_greedy | -0.291370 +/- 0.139473 | 1.0000 | 2.6712 | 32.6933 | 0.01748 |
| latency_greedy | -7.870699 +/- 25.907972 | 0.9271 | 7.7744 | 41.8919 | 0.01748 |
| dros_pure | -11.449900 +/- 31.114318 | 0.8906 | 16.3586 | 48.5792 | 0.49699 |
| block_balanced | -13.923657 +/- 33.393541 | 0.8698 | 12.4982 | 46.9756 | 0.01748 |
| random | -100.000000 +/- 0.000000 | 0.0000 | 111.1396 | 106.0901 | 0.01748 |

DROS-pure margin vs best non-DROS: mean `-11.25093412`, min `-99.86165959`, win/tie rate `0.0260`.
DROS-hybrid margin vs best non-DROS: mean `0.00000000`, min `0.00000000`, win/tie rate `1.0000`.
Autoreg-RL-pure margin vs best non-DROS: mean `0.00168536`, min `-0.12469752`, win/tie rate `0.5208`.
