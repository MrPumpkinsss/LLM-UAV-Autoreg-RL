# Real-Profile k Sweep Benchmark

Fairness: all `autoreg_rl_pure_k*` rows use only learned-policy candidates plus generic feasibility projection. The strong heuristic actions are benchmark baselines only, not RL candidate-pool entries.

States: `320` (`91,92,93,94,95` x `64`)
Profile: `Qwen3-0.6B real profile`
Policy: `results/autoreg_rl_real_k16_blocks/autoreg_policy_best.pt`

| method | reward | feasible | latency | PPL | runtime_s | margin | win/tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure_k256 | -0.255559 +/- 0.066184 | 1.0000 | 2.4919 | 31.3157 | 0.06565 | 0.017895 | 0.9500 |
| autoreg_rl_pure_k64 | -0.257609 +/- 0.067512 | 1.0000 | 2.5106 | 31.3289 | 0.03110 | 0.015845 | 0.8625 |
| autoreg_rl_pure_k16 | -0.263797 +/- 0.074131 | 1.0000 | 2.5672 | 31.3703 | 0.02796 | 0.009656 | 0.6687 |
| baseline:hybrid_heuristic | -0.273453 +/- 0.086901 | 1.0000 | 2.6411 | 31.5446 | 0.01120 |  |  |
| baseline:beam_search | -0.278149 +/- 0.094707 | 1.0000 | 2.6786 | 31.6178 | 0.01120 |  |  |
| baseline:simulated_annealing | -0.351507 +/- 0.146873 | 1.0000 | 3.2171 | 33.1205 | 0.01120 |  |  |
| baseline:local_search | -0.352602 +/- 0.148446 | 1.0000 | 3.2242 | 33.1506 | 0.01120 |  |  |
| baseline:pdp_aware_greedy | -0.369239 +/- 0.178122 | 1.0000 | 3.3570 | 33.4093 | 0.01120 |  |  |
| baseline:latency_greedy | -6.340535 +/- 22.905382 | 0.9437 | 8.3329 | 43.5884 | 0.01120 |  |  |
| baseline:block_balanced | -13.781014 +/- 33.106199 | 0.8719 | 12.7195 | 48.6002 | 0.01120 |  |  |
| baseline:random | -100.000000 +/- 0.000000 | 0.0000 | 109.9627 | 102.5448 | 0.01120 |  |  |

## RL Candidate Budget

| k | mean margin | min margin | win/tie | strict win | runtime_s |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.00965595 | -0.17289444 | 0.6687 | 0.3531 | 0.02796 |
| 64 | 0.01584474 | -0.05711514 | 0.8625 | 0.4562 | 0.03110 |
| 256 | 0.01789450 | -0.00867147 | 0.9500 | 0.5125 | 0.06565 |
