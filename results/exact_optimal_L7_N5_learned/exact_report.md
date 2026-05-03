# Exact Optimal Comparison

Exact exhaustive optimal comparison is only tractable for reduced toy instances.
This run uses `7` layers and `5` UAVs, so each state has `78125` assignments.

Policy loaded: `True`

| method | reward | feasible | gap_mean | gap_max | latency | PPL |
|---|---:|---:|---:|---:|---:|---:|
| best_heuristic | -0.019228 +/- 0.002058 | 1.0000 | 0.000000 | 0.000000 | 0.1923 | 30.8247 |
| optimal | -0.019228 +/- 0.002058 | 1.0000 | 0.000000 | 0.000000 | 0.1923 | 30.8247 |
| autoreg_rl_pure | -0.019522 +/- 0.002174 | 1.0000 | 0.000294 | 0.006520 | 0.1952 | 30.8247 |
