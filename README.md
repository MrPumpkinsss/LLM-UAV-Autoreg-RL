# LLM-UAV Autoregressive RL

Clean experiment repository for UAV-enabled distributed LLM submodel deployment.

The main result is a teacher-warm-start autoregressive RL policy for Qwen3-0.6B profile-based simulation. At inference time, `autoreg_rl_pure` samples deployment candidates only from the learned policy and applies generic feasibility projection. It does not insert beam-search, local-search, simulated-annealing, or greedy heuristic actions into the RL candidate pool.

## Current Result

Final report: `results/autoreg_rl_strong_benchmark_report.md`

Latest full benchmark: `3 seeds x 64 states = 192 states`, `N=5` UAVs, `28` LLM layers, `k=64` RL candidates per state, evaluated on CUDA.

Reward is the negative weighted cost for feasible deployments:

```text
reward = -cost
cost = alpha * ((PPL_hat - PPL_ref) / PPL_ref) + beta * (latency_s / latency_ref_s)
```

Hard-infeasible deployments receive `reward = -100.0`. The calibrated config uses `alpha = 0.4`, `beta = 0.6`, `latency_ref_s = 6.0`, and `PPL_ref = 30.824672`.

| method | reward | feasible | latency | PPL | runtime/state |
|---|---:|---:|---:|---:|---:|
| hybrid_heuristic | -0.199647 +/- 0.069228 | 1.0000 | 1.9431 | 31.2361 | 0.01075s |
| beam_search | -0.201854 +/- 0.072310 | 1.0000 | 1.9615 | 31.2646 | 0.01075s |
| autoreg_rl_pure | -0.212208 +/- 0.076782 | 1.0000 | 2.0500 | 31.3805 | 0.02837s |
| simulated_annealing | -0.267850 +/- 0.134952 | 1.0000 | 2.4852 | 32.3142 | 0.01075s |
| local_search | -0.269031 +/- 0.136775 | 1.0000 | 2.4911 | 32.3601 | 0.01075s |
| pdp_aware_greedy | -0.286648 +/- 0.154524 | 1.0000 | 2.6246 | 32.6885 | 0.01075s |
| latency_greedy | -8.859663 +/- 27.554457 | 0.9167 | 10.4054 | 41.0712 | 0.01075s |
| block_balanced | -16.508051 +/- 36.025838 | 0.8438 | 18.0614 | 47.5062 | 0.01075s |
| random | -100.000000 +/- 0.000000 | 0.0000 | 107.6689 | 100.4916 | 0.01075s |

Mean margin of `autoreg_rl_pure` vs the best non-RL heuristic: `-0.01256090`. Win/tie rate: `0.3906`.

With `k=64`, the learned policy remains feasible but no longer beats the strongest heuristic on mean reward.

## Visuals

![Training curves](results/visuals_autoreg/training_curves.png)

![Benchmark reward comparison](results/visuals_autoreg/benchmark_reward_bar.png)

![Benchmark feasibility comparison](results/visuals_autoreg/benchmark_feasibility_bar.png)

![RL margin histogram](results/visuals_autoreg/margin_histogram.png)

![RL margin by seed](results/visuals_autoreg/margin_by_seed.png)

![Autoreg-RL vs best heuristic](results/visuals_autoreg/autoreg_vs_heuristic_scatter.png)

## Real LLM Benchmark

The real-model check loads `Qwen/Qwen3-0.6B` with `bfloat16` on CUDA and measures the profile used by the simulator. The full baseline comparison then uses the measured layer parameter counts, reference PPL, and fitted corruption curve in the LLM-UAV environment.

Profile results are stored in `results/qwen3_0p6b_real_profile`. Full baseline benchmark results are stored in `results/benchmark_real_profile_k64`.

| item | value |
|---|---:|
| model | `Qwen/Qwen3-0.6B` |
| parameters | 596,049,920 |
| layers | 28 |
| hidden size | 1024 |
| layer params mean | 15,730,944 |
| dtype | bfloat16 |
| forward latency mean | 38.422 ms |
| reference PPL | 30.824672 |
| fitted gamma | 10.899648 |
| surrogate fit R2 | 0.997366 |
| log-ratio RMSE | 0.019277 |

Real-profile baseline comparison, `3 seeds x 64 states`, `k=64`:

| method | reward | feasible | latency | PPL | runtime/state |
|---|---:|---:|---:|---:|---:|
| hybrid_heuristic | -0.269523 +/- 0.082058 | 1.0000 | 2.6010 | 31.5508 | 0.01192s |
| beam_search | -0.273802 +/- 0.087047 | 1.0000 | 2.6352 | 31.6174 | 0.01192s |
| autoreg_rl_pure | -0.289838 +/- 0.128823 | 1.0000 | 2.7729 | 31.7920 | 0.03171s |
| simulated_annealing | -0.344778 +/- 0.133848 | 1.0000 | 3.1465 | 33.1463 | 0.01192s |
| local_search | -0.345173 +/- 0.133750 | 1.0000 | 3.1503 | 33.1472 | 0.01192s |
| pdp_aware_greedy | -0.357012 +/- 0.144814 | 1.0000 | 3.2462 | 33.3209 | 0.01192s |
| latency_greedy | -4.895185 +/- 19.886923 | 0.9583 | 8.3558 | 43.4784 | 0.01192s |
| block_balanced | -10.417073 +/- 28.891749 | 0.9062 | 12.6621 | 48.6316 | 0.01192s |
| random | -100.000000 +/- 0.000000 | 0.0000 | 110.8475 | 108.7251 | 0.01192s |

Mean margin of `autoreg_rl_pure` vs the best non-RL heuristic on the real-profile benchmark: `-0.02031545`. Win/tie rate: `0.2708`.

### Best Real-Profile k=16 Block Policy

The strongest current learned policy uses the real Qwen3-0.6B profile, reward-weighted teacher warm start, a pure RL block projection decoder, and a pure policy beam candidate generator. Inference still uses only learned-policy candidates; no baseline beam-search, local-search, simulated-annealing, or greedy heuristic actions are inserted into the RL candidate pool.

Checkpoint: `results/autoreg_rl_real_k16_blocks/autoreg_policy_best.pt`
Benchmark: `results/benchmark_real_profile_k16_blocks_policy_beam`

Real-profile baseline comparison, `3 seeds x 64 states`, `k=16`, `max_blocks=5`, `candidate_mode=beam`:

| method | reward | feasible | latency | PPL | runtime/state |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.261002 +/- 0.072367 | 1.0000 | 2.5350 | 31.4028 | 0.01985s |
| hybrid_heuristic | -0.269523 +/- 0.082058 | 1.0000 | 2.6010 | 31.5508 | 0.01045s |
| beam_search | -0.273802 +/- 0.087047 | 1.0000 | 2.6352 | 31.6174 | 0.01045s |
| simulated_annealing | -0.344778 +/- 0.133848 | 1.0000 | 3.1465 | 33.1463 | 0.01045s |
| local_search | -0.345173 +/- 0.133750 | 1.0000 | 3.1503 | 33.1472 | 0.01045s |
| pdp_aware_greedy | -0.357012 +/- 0.144814 | 1.0000 | 3.2462 | 33.3209 | 0.01045s |

Mean margin of `autoreg_rl_pure` vs the best non-RL heuristic: `0.00852120`. Win/tie rate: `0.6406`.

Reproduce it with:

```powershell
python -m src.benchmark_real_profile `
  --config configs/qwen3_calibrated.yaml `
  --real-dir results/qwen3_0p6b_real_profile `
  --policy results/autoreg_rl_real_k16_blocks/autoreg_policy_best.pt `
  --states 64 `
  --seeds "91,92,93" `
  --beam-width 32 `
  --anneal-steps 128 `
  --autoreg-candidates 16 `
  --autoreg-refine-steps 0 `
  --projection-mode blocks `
  --max-blocks 5 `
  --candidate-mode beam `
  --beam-temperature 1.0 `
  --out results/benchmark_real_profile_k16_blocks_policy_beam `
  --device cuda
```

Reproduce it with:

```powershell
python -m src.real_llm_profile --config configs/real_llm.yaml
python -m src.benchmark_real_profile `
  --config configs/qwen3_calibrated.yaml `
  --real-dir results/qwen3_0p6b_real_profile `
  --policy results/autoreg_rl_teacher/autoreg_policy_best.pt `
  --states 64 `
  --seeds "91,92,93" `
  --beam-width 32 `
  --anneal-steps 128 `
  --autoreg-candidates 64 `
  --out results/benchmark_real_profile_k64 `
  --device cuda
```

## Surrogate Benchmark

The surrogate benchmark is placed last because it validates the PPL model used by both the surrogate and real-profile deployment experiments. It compares the exponential surrogate `PPL_hat = PPL_ref * exp(gamma * damage)` against measured Qwen3-0.6B PPL under controlled embedding corruption.

Results are stored in `results/surrogate_benchmark`.

![Surrogate PPL fit](results/surrogate_benchmark/surrogate_ppl_fit.png)

| metric | value |
|---|---:|
| PPL_ref | 30.824672 |
| gamma | 10.899648 |
| R2 log-ratio | 0.997366 |
| RMSE log-ratio | 0.019277 |
| MAE PPL | 0.796831 |
| Max abs PPL error | 1.587679 |
| Mean relative PPL error | 0.015270 |
| Max relative PPL error | 0.030786 |

## Repository Layout

- `src/autoreg_rl_agent.py`: masked autoregressive policy for layer-to-UAV assignment.
- `src/train_autoreg_rl.py`: teacher warm-start plus RL/self-imitation training.
- `src/benchmark_autoreg.py`: fair benchmark against strong heuristics.
- `src/exact_optimal_compare.py`: exact exhaustive comparison on reduced 5-UAV instances.
- `src/baselines.py`: greedy, local search, beam search, simulated annealing, and hybrid heuristic.
- `src/env.py`: LLM-UAV simulator, constraints, reward, KKT bandwidth allocation.
- `src/real_llm_profile.py`: Qwen3-0.6B profile/PPL calibration script.
- `configs/qwen3_calibrated.yaml`: calibrated simulation config.
- `results/autoreg_rl_teacher/autoreg_policy_best.pt`: best policy checkpoint.
- `results/benchmark_autoreg_gpu_k64`: latest full CUDA benchmark with `k=64`.
- `results/visuals_autoreg`: generated figures embedded above.

## Setup

```powershell
conda create -n LLM-UAV python=3.11 -y
conda activate LLM-UAV
pip install numpy pandas pyyaml matplotlib transformers datasets
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Use the existing local environment if it already exists:

```powershell
conda activate LLM-UAV
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Reproduce Benchmark

```powershell
python -m src.benchmark_autoreg `
  --config configs/qwen3_calibrated.yaml `
  --policy results/autoreg_rl_teacher/autoreg_policy_best.pt `
  --states 64 `
  --seeds "91,92,93" `
  --beam-width 32 `
  --anneal-steps 128 `
  --autoreg-candidates 64 `
  --autoreg-refine-steps 0 `
  --out results/benchmark_autoreg_repro `
  --device cuda
```

## Exact Optimal Check

The full `28 layers x 5 UAV` problem is too large for exhaustive search. For a true optimality check, keep `N=5` and reduce the number of layers:

```powershell
python -m src.exact_optimal_compare `
  --config configs/qwen3_calibrated.yaml `
  --policy results/autoreg_rl_exact_L7_N5/autoreg_policy_best.pt `
  --out results/exact_optimal_L7_N5_repro `
  --num-layers 7 `
  --num-uavs 5 `
  --states 64 `
  --max-states 1000000 `
  --autoreg-candidates 2048
```

The checked run in `results/exact_optimal_L7_N5_learned` enumerates `5^7 = 78125` assignments per state. In that reduced 5-UAV setting, the strongest heuristic matches the exact optimum, while `autoreg_rl_pure` remains close with mean optimality gap `0.000294`.

## Train

```powershell
python -m src.train_autoreg_rl `
  --config configs/qwen3_calibrated.yaml `
  --out results/autoreg_rl_teacher_repro `
  --teacher-states 1500 `
  --teacher-updates 1200 `
  --episodes 300 `
  --batch-states 8 `
  --candidates 128 `
  --validation-states 64 `
  --eval-candidates 512 `
  --device cuda
```

## Fairness Note

The RL policy was trained with teacher warm-start from strong heuristic solutions. This should be reported as teacher-warm-start RL. During inference/benchmarking, heuristic actions are not placed into the RL candidate pool.
