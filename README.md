# LLM-UAV Autoregressive RL

This repository implements UAV-enabled distributed LLM layer placement with an autoregressive RL policy. The environment uses `N=5` UAVs, hard memory and energy constraints, KKT closed-form bandwidth allocation, retransmission-aware residual packet loss, and a real-LLM-calibrated PPL surrogate.

At inference time, `autoreg_rl_pure` samples candidate layer assignments only from the learned policy and applies generic feasibility projection. Strong heuristics are benchmark baselines only; their actions are not inserted into the RL candidate pool.

## Current Status

Three model profiles are tracked:

| model | status | policy checkpoint | benchmark |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | main validated result | `results/autoreg_rl_layer_calibrated_hard_k256/autoreg_policy_best.pt` | `results/benchmark_layer_calibrated_hard_k256_5seed` |
| `Qwen/Qwen3.5-4B` | larger-model experimental result | `results/autoreg_rl_qwen35_4b_v2_teacher_big/autoreg_policy_best.pt` | `results/benchmark_qwen35_4b_v2_teacher_big_5seed` |
| `google/gemma-4-E4B` | base-model exploratory result | `results/autoreg_rl_gemma4_e4b_teacher/autoreg_policy_best.pt` | `results/benchmark_gemma4_e4b_teacher` |

The Gemma result is useful as a policy/runtime experiment, but its PPL surrogate is weaker than the Qwen surrogates, so Qwen3-0.6B remains the strongest paper-ready line.

## Observation, Action, Reward

For Qwen3-0.6B, `N=5` and `L=28`, so the flattened base observation is `R^93`:

| component | shape | dim | normalization |
|---|---:|---:|---|
| SNR matrix | `N x N` | 25 | `log1p(snr) / log1p(1e6)` |
| packet-drop probability matrix | `N x N` | 25 | raw probability |
| UAV compute capacity | `N` | 5 | `compute_hz / 1e10` |
| UAV memory capacity | `N` | 5 | `mem_bytes / 512 MiB` |
| UAV energy budget | `N` | 5 | `energy_j / 2000` |
| previous layer placement | `L` | 28 | `previous_uav_id / (N - 1)` |

At each autoregressive step, the policy adds layer features and partial-assignment context:

| step input | dim |
|---|---:|
| encoded base observation | 512 |
| layer features: memory, cycles, previous activation, next activation, importance, position | 6 |
| previous assigned UAV one-hot | 5 |
| normalized memory used per UAV | 5 |
| layer index fraction | 1 |
| remaining memory fraction | 1 |

The action is a layer-to-UAV vector:

```text
a = [u_0, u_1, ..., u_{L-1}], where u_l in {0, 1, 2, 3, 4}
```

For feasible deployments:

```text
reward = -cost
cost = alpha * ((PPL_hat - PPL_ref) / PPL_ref) + beta * (latency_s / latency_ref_s)
```

The released configs use `alpha = 0.4`, `beta = 0.6`, one retransmission, and `infeasible_reward = -100.0`. The retransmission setting affects both expected communication latency and the residual activation-loss probability:

```text
residual_l = p_l^(r + 1)
```

With `r = 1`, PPL damage uses `p_l^2` rather than raw PDP.

## RL Method

Training uses policy sampling, environment scoring, policy-gradient updates, and self-imitation replay. Larger runs use teacher warm-start: strong heuristics generate replay targets before RL continues optimizing the same reward. This improves learning but does not change inference fairness, because benchmark-time `autoreg_rl_pure` still uses only learned-policy candidates.

The default high-quality inference setting uses beam policy candidates with `k=256`.

## Surrogate Principle

The paper-formula PPL is:

```text
PPL = exp(-1/M * sum_k log P_LLM(w_k | w_1, ..., w_{k-1}))
```

The simulator uses a calibrated surrogate:

```text
PPL_hat = PPL_ref * exp(sum_l gamma_l * residual_l)
residual_l = p_l^(r + 1)
```

A boundary contributes PPL damage only when adjacent layers are placed on different UAVs. The layer sensitivity `gamma_l` is fitted by loading the real LLM, corrupting hidden states at controlled drop rates, and measuring real PPL.

## Simulator Benchmarks

These are the main policy-comparison benchmarks. PPL is `PPL_hat`, not an online real-LLM call.

| model | artifact | states | RL reward | best non-RL reward | latency | PPL_hat | RL runtime | mean margin | win/tie |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | `results/benchmark_layer_calibrated_hard_k256_5seed` | 320 | -0.497273 | -0.499117 | 2.6278 | 48.8752 | 0.03475 | +0.001844 | 0.8125 |
| Qwen3.5-4B | `results/benchmark_qwen35_4b_v2_teacher_big_5seed` | 80 | -0.146974 | -0.147090 | 4.4509 | 12.8052 | 0.05259 | +0.000116 | 0.8250 |
| Gemma-4-E4B | `results/benchmark_gemma4_e4b_teacher` | 192 | -0.277215 | -0.277479 | 9.1506 | 10.8171 | 0.07353 | +0.000264 | 0.7396 |

Strong non-RL baselines include `hybrid_heuristic`, `block_lns_strong`, `block_beam_strong`, `beam_search`, `simulated_annealing`, `local_search`, `pdp_aware_greedy`, `latency_greedy`, `block_balanced`, and `random`.

### Qwen3-0.6B Methods

| method | reward | feasible | latency | PPL_hat | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.497273 | 1.0000 | 2.6278 | 48.8752 | 0.03475 |
| hybrid_heuristic | -0.499117 | 1.0000 | 2.6578 | 48.7862 | 0.01226 |
| block_lns_strong | -0.499191 | 1.0000 | 2.6519 | 48.8368 | 0.06524 |
| block_beam_strong | -0.584792 | 1.0000 | 2.7353 | 54.7887 | 0.30765 |
| beam_search | -17.326284 | 1.0000 | 2.8847 | 1343.2341 | 0.03031 |

### Qwen3.5-4B Methods

| method | reward | feasible | latency | PPL_hat | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.146974 | 1.0000 | 4.4509 | 12.8052 | 0.05259 |
| hybrid_heuristic | -0.147090 | 1.0000 | 4.4668 | 12.7940 | 0.01355 |
| block_lns_strong | -0.147090 | 1.0000 | 4.4668 | 12.7940 | 0.06099 |
| block_beam_strong | -0.148220 | 1.0000 | 4.4904 | 12.8071 | 0.66905 |
| beam_search | -3.568186 | 1.0000 | 4.7848 | 118.4562 | 0.06955 |

### Gemma-4-E4B Methods

| method | reward | feasible | latency | PPL_hat | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.277215 | 1.0000 | 9.1506 | 10.8171 | 0.07353 |
| hybrid_heuristic | -0.277479 | 1.0000 | 9.1666 | 10.8113 | 0.01737 |
| block_lns_strong | -0.277479 | 1.0000 | 9.1666 | 10.8113 | 0.05510 |
| block_beam_strong | -0.280392 | 1.0000 | 9.2573 | 10.8165 | 0.49193 |
| beam_search | -0.301616 | 1.0000 | 9.9213 | 10.8515 | 0.09442 |

## Real LLM Validation

Real LLM validation recomputes paper-formula PPL for selected deployment actions. It is slower than simulator benchmarking and is used to validate the surrogate.

| model | artifact | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | `results/real_action_ppl_validation_layer_calibrated_retrained` | 64 competitive | 0.023323 | 0.216265 | 0.052984 | 0.996233 | 0.973764 | validated |
| Qwen3.5-4B | not available | 0 | - | - | - | - | - | pending |
| Gemma-4-E4B | not available | 0 | - | - | - | - | - | pending |

## Surrogate Benchmarks

Surrogate quality is reported separately for each model.

| model | surrogate artifact | clean PPL | fit type | R2 | RMSE log-ratio | mean rel error | max rel error | interpretation |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | `results/surrogate_benchmark_qwen3_0p6b` | 30.811979 | embedding exponential + layer gamma | 0.997363 | 0.019289 | 0.015285 | 0.030760 | strong |
| Qwen3.5-4B | `results/surrogate_benchmark_qwen35_4b_v2` | 12.388740 | layer one-hot MLP | 0.999725 | 0.001998 | - | - | strongest surrogate fit |
| Gemma-4-E4B | `results/surrogate_benchmark_gemma4_e4b` | 10.744652 | embedding exponential | 0.366344 | 0.101227 | 0.058760 | 0.139512 | weak, use cautiously |

Qwen3.5 also has a high-quality layer-wise analytic calibration: layer gamma sum `206.094048`, mean layer R2 `0.995787`, and layer RMSE log-ratio `0.006596`. Gemma currently has no full layer-wise calibration in the tracked result set; its simulator benchmark is therefore exploratory. A compact cross-model surrogate report is stored at `results/surrogate_benchmark_multi_model_report.md`.

## Visuals

### Qwen3-0.6B

![0.6B training curves](results/visuals_layer_calibrated_hard_k256/training_curves.png)

![0.6B benchmark reward comparison](results/visuals_layer_calibrated_hard_k256/benchmark_reward_bar.png)

![0.6B RL margin histogram](results/visuals_layer_calibrated_hard_k256/margin_histogram.png)

### Qwen3.5-4B

![Qwen3.5 training curves](results/visuals_qwen35_teacher_big/training_curves.png)

![Qwen3.5 benchmark reward comparison](results/visuals_qwen35_teacher_big/benchmark_reward_bar.png)

![Qwen3.5 RL margin histogram](results/visuals_qwen35_teacher_big/margin_histogram.png)

### Surrogate

![Qwen3-0.6B surrogate fit](results/surrogate_benchmark_qwen3_0p6b/surrogate_ppl_fit.png)

![Gemma-4-E4B surrogate fit](results/surrogate_benchmark_gemma4_e4b/surrogate_ppl_fit.png)

## Reproduce

Train Qwen3-0.6B:

```powershell
python -m src.train_autoreg_rl `
  --config configs/qwen3_calibrated.yaml `
  --out results/autoreg_rl_qwen3_0p6b_teacher_big `
  --device cuda `
  --episodes 2000 `
  --batch-states 32 `
  --candidates 256 `
  --eval-candidates 256 `
  --teacher-states 1000 `
  --teacher-updates 1000 `
  --projection-mode blocks `
  --max-blocks 5 `
  --candidate-mode beam
```

Benchmark a trained policy:

```powershell
python -m src.benchmark_real_profile `
  --config configs/qwen3_calibrated.yaml `
  --policy results/autoreg_rl_layer_calibrated_hard_k256/autoreg_policy_best.pt `
  --states 64 `
  --seeds "91,92,93,94,95" `
  --beam-width 32 `
  --anneal-steps 128 `
  --autoreg-candidates 256 `
  --out results/benchmark_layer_calibrated_hard_k256_5seed `
  --device cuda
```

Run a surrogate benchmark:

```powershell
python -m src.benchmark_surrogate `
  --real-dir results/qwen3_0p6b_real_profile `
  --out results/surrogate_benchmark_qwen3_0p6b
```

## Repository Layout

```text
.
|-- configs/
|   |-- qwen3_calibrated.yaml
|   |-- qwen35_4b_calibrated.yaml
|   |-- gemma4_base_calibrated.yaml
|   |-- real_llm.yaml
|   |-- real_llm_qwen35_4b.yaml
|   `-- real_llm_gemma4_base.yaml
|-- src/
|   |-- env.py
|   |-- channel.py
|   |-- llm_profile.py
|   |-- autoreg_rl_agent.py
|   |-- train_autoreg_rl.py
|   |-- baselines.py
|   |-- benchmark_real_profile.py
|   |-- benchmark_real_action_ppl.py
|   |-- benchmark_surrogate.py
|   |-- real_llm_profile.py
|   |-- real_llm_layer_calibration.py
|   `-- make_visuals.py
|-- results/
|   |-- qwen3_0p6b_real_profile/
|   |-- qwen35_4b_real_profile_v2/
|   |-- gemma4_e4b_real_profile/
|   |-- autoreg_rl_layer_calibrated_hard_k256/
|   |-- autoreg_rl_qwen35_4b_v2_teacher_big/
|   |-- autoreg_rl_gemma4_e4b_teacher/
|   |-- benchmark_layer_calibrated_hard_k256_5seed/
|   |-- benchmark_qwen35_4b_v2_teacher_big_5seed/
|   |-- benchmark_gemma4_e4b_teacher/
|   |-- surrogate_benchmark_qwen3_0p6b/
|   |-- surrogate_benchmark_gemma4_e4b/
|   `-- visuals_layer_calibrated_hard_k256/
|-- requirements.txt
`-- README.md
```

Generated experiment outputs are ignored by default. The repository tracks only curated checkpoints, profile files, benchmark reports, validation reports, and README figures.

## Setup

```powershell
conda create -n LLM-UAV python=3.11 -y
conda activate LLM-UAV
pip install numpy pandas pyyaml matplotlib transformers datasets
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
