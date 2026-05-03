# Autoregressive Pure RL Strong Benchmark

Inference fairness: `autoreg_rl_pure` uses only actions sampled from the autoregressive RL policy plus generic feasibility projection. No beam, local-search, simulated-annealing, or greedy heuristic action is inserted into its candidate pool.

Training note: this checkpoint uses teacher warm-start from strong heuristic solutions, followed by RL/self-imitation updates. This is not a heuristic candidate pool at inference time, but it should be described as teacher-warm-start RL.

## Training

- Train directory: `results/autoreg_rl_teacher`
- Teacher reward mean: `-0.204520`
- Best validation episode: `250`
- Best validation reward: `-0.203422`
- Best validation feasible rate: `1.0000`
- Checkpoint: `results/autoreg_rl_teacher/autoreg_policy_best.pt`

## Benchmark

- Benchmark directory: `results/benchmark_autoreg_gpu_k64`
- States: `192`

| method | reward | feasible | latency | PPL | runtime_s |
|---|---:|---:|---:|---:|---:|
| hybrid_heuristic | -0.199647 +/- 0.069228 | 1.0000 | 1.9431 | 31.2361 | 0.01075 |
| beam_search | -0.201854 +/- 0.072310 | 1.0000 | 1.9615 | 31.2646 | 0.01075 |
| autoreg_rl_pure | -0.212208 +/- 0.076782 | 1.0000 | 2.0500 | 31.3805 | 0.02837 |
| simulated_annealing | -0.267850 +/- 0.134952 | 1.0000 | 2.4852 | 32.3142 | 0.01075 |
| local_search | -0.269031 +/- 0.136775 | 1.0000 | 2.4911 | 32.3601 | 0.01075 |
| pdp_aware_greedy | -0.286648 +/- 0.154524 | 1.0000 | 2.6246 | 32.6885 | 0.01075 |
| latency_greedy | -8.859663 +/- 27.554457 | 0.9167 | 10.4054 | 41.0712 | 0.01075 |
| block_balanced | -16.508051 +/- 36.025838 | 0.8438 | 18.0614 | 47.5062 | 0.01075 |
| random | -100.000000 +/- 0.000000 | 0.0000 | 107.6689 | 100.4916 | 0.01075 |

## Result

- `autoreg_rl_pure` margin vs best non-RL heuristic: `-0.01256090` mean, `-0.24414838` min.
- `autoreg_rl_pure` win/tie rate vs best non-RL heuristic: `0.3906`.
- Under this benchmark, autoregressive pure RL does not exceed the strongest heuristic on mean reward.
- `dros_hybrid` remains an upper-reference hybrid method and should not be called pure RL.
