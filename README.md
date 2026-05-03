# LLM-UAV Autoregressive RL

Clean experiment repository for UAV-enabled distributed LLM submodel deployment.

The main result is a teacher-warm-start autoregressive RL policy for Qwen3-0.6B profile-based simulation. At inference time, `autoreg_rl_pure` samples deployment candidates only from the learned policy and applies generic feasibility projection. It does not insert beam-search, local-search, simulated-annealing, or greedy heuristic actions into the RL candidate pool.

## Result

Final report: `results/autoreg_rl_strong_benchmark_report.md`

Benchmark: `3 seeds x 64 states = 192 states`, `N=5` UAVs.

| method | reward | feasible | latency | PPL |
|---|---:|---:|---:|---:|
| autoreg_rl_pure | -0.197280 +/- 0.058722 | 1.0000 | 1.9311 | 31.1460 |
| hybrid_heuristic | -0.198966 +/- 0.057822 | 1.0000 | 1.9493 | 31.1356 |
| beam_search | -0.201209 +/- 0.062074 | 1.0000 | 1.9696 | 31.1523 |

Mean margin of `autoreg_rl_pure` vs the best non-RL heuristic: `+0.00168536`.

## Repository Layout

- `src/autoreg_rl_agent.py`: masked autoregressive policy for layer-to-UAV assignment.
- `src/train_autoreg_rl.py`: teacher warm-start plus RL/self-imitation training.
- `src/benchmark_autoreg.py`: fair benchmark against strong heuristics.
- `src/baselines.py`: greedy, local search, beam search, simulated annealing, and hybrid heuristic.
- `src/env.py`: LLM-UAV simulator, constraints, reward, KKT bandwidth allocation.
- `src/real_llm_profile.py`: Qwen3-0.6B profile/PPL calibration script.
- `configs/qwen3_calibrated.yaml`: calibrated simulation config.
- `results/autoreg_rl_teacher/autoreg_policy_best.pt`: best policy checkpoint.
- `results/benchmark_autoreg_1024_3seed`: final benchmark summary.

## Setup

```powershell
conda create -n LLM-UAV python=3.11 -y
conda activate LLM-UAV
pip install numpy pandas pyyaml torch transformers datasets
```

Use the existing local environment if it already exists:

```powershell
conda activate LLM-UAV
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
  --autoreg-candidates 1024 `
  --out results/benchmark_autoreg_repro
```

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

