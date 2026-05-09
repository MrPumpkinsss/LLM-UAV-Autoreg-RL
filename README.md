# LLM-UAV Autoregressive RL

This repository implements UAV-enabled distributed LLM layer placement with an autoregressive RL policy. The environment uses `N=5` UAVs, hard memory and energy constraints, KKT closed-form bandwidth allocation, retransmission-aware residual packet loss, and a real-LLM-calibrated PPL surrogate.

At inference time, `autoreg_rl_pure` samples candidate layer assignments only from the learned policy and applies generic feasibility projection. Strong heuristics are benchmark baselines only; their actions are not inserted into the RL candidate pool.

## Current Status

Three model profiles are tracked:

| model | status | policy checkpoint | benchmark |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | main validated result | `results/autoreg_rl_layer_calibrated_hard_k256/autoreg_policy_best.pt` | `results/benchmark_layer_calibrated_hard_k256_5seed` |
| `Qwen/Qwen3.5-4B` | larger-model experimental result | `results/autoreg_rl_qwen35_4b_v2_teacher_big/autoreg_policy_best.pt` | `results/benchmark_qwen35_4b_v2_teacher_big_5seed` |
| `google/gemma-4-E4B` | base-model exploratory result with curve surrogate | `results/autoreg_rl_gemma4_e4b_teacher/autoreg_policy_best.pt` | `results/benchmark_gemma4_e4b_teacher` |

Gemma now uses a denser real-LLM embedding-drop profile and an empirical piecewise curve surrogate. Qwen3-0.6B remains the most completely validated line because it has layer-wise calibration and real action-level validation. With pure learned-policy candidates and overgenerate-then-deduplicate beam inference, all three tracked simulator benchmarks now beat the strongest non-RL baseline on mean reward.

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

The main `r = 1` configs use `alpha = 0.4`, `beta = 0.6`, linear `PPL_norm`, one retransmission, and `infeasible_reward = -100.0`. The Qwen3 `r = 0` rerun switches the PPL term to `log(1 + PPL_norm)` because raw PDP otherwise creates a heavy-tail reward. The retransmission setting affects both expected communication latency and the residual activation-loss probability:

```text
residual_l = p_l^(r + 1)
```

With `r = 1`, PPL damage uses `p_l^2` rather than raw PDP.

## RL Method

Training uses policy sampling, environment scoring, policy-gradient updates, and self-imitation replay. Larger runs use teacher warm-start: strong heuristics generate replay targets before RL continues optimizing the same reward. This improves learning but does not change inference fairness, because benchmark-time `autoreg_rl_pure` still uses only learned-policy candidates.

The default high-quality inference setting uses beam policy candidates with `k=256`; Qwen3-0.6B and Qwen3.5-4B overgenerate raw policy beams and deduplicate them before scoring. This is still pure RL inference because no heuristic action is inserted into the candidate pool.

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

For Gemma-4-E4B, the single exponential baseline is not expressive enough because sampled corruption trials produce a noisy non-linear curve. Gemma therefore uses:

```text
damage = sum_l importance_l * residual_l
PPL_hat = PPL_ref * exp(f_curve(damage))
```

where `f_curve` is an empirical piecewise interpolation fitted from the real Gemma PPL corruption curve. The report still records the old exponential baseline R2 separately.

This is a curve fit on sampled corruption points, not a layer-wise analytic calibration.

## Simulator Benchmarks

These are the main policy-comparison benchmarks. PPL is `PPL_hat`, not an online real-LLM call. The tables below are generated by the commands in [Reproduce](#reproduce). Runtime is machine-dependent; reward, feasibility, latency, and PPL should match the tracked `benchmark_rows.csv` files.

### Retransmission-Aware Main Benchmark (`r = 1`)

| model | artifact | states | RL reward | best non-RL | best non-RL reward | RL latency | RL PPL_hat | RL runtime | mean margin | win/tie |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | `results/benchmark_layer_calibrated_hard_k256_5seed` | 320 | -0.571681 | `hybrid_heuristic` | -0.591107 | 2.6322 | 54.5724 | 0.21232 | +0.019426 | 0.6813 |
| Qwen3.5-4B | `results/benchmark_qwen35_4b_v2_teacher_big_5seed` | 80 | -0.164647 | `hybrid_heuristic` | -0.164903 | 4.4425 | 13.3604 | 0.11292 | +0.000256 | 0.9250 |
| Gemma-4-E4B | `results/benchmark_gemma4_e4b_teacher` | 192 | -0.278169 | `hybrid_heuristic` | -0.278862 | 9.2066 | 10.7976 | 0.10848 | +0.000693 | 0.7552 |

Strong non-RL baselines include `hybrid_heuristic`, `block_lns_strong`, `block_beam_strong`, `beam_search`, `simulated_annealing`, `local_search`, `pdp_aware_greedy`, `latency_greedy`, `block_balanced`, and `random`.

### No-Retransmission Ablation (`r = 0`)

The no-retransmission ablation sets `wireless.retransmissions = 0`, so residual activation loss is raw PDP instead of `PDP^2`. These reruns use `reward.ppl_cost_mode = log`, so the reward uses `log(1 + PPL_norm)` instead of raw `PPL_norm`. This avoids the pathological heavy-tail explosion seen with the old linear-cost no-retrans run. These runs were executed in the `LLM-UAV` conda environment with CUDA.

| model | artifact | states | RL reward | best non-RL | best non-RL reward | RL latency | RL PPL_hat | RL runtime | mean margin | win/tie |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | `results/benchmark_qwen3_0p6b_no_retrans_logcost_hard_5seed` | 320 | -0.257262 | `hybrid_heuristic` | -0.259101 | 2.3714 | 32.4630 | 0.11442 | +0.001839 | 0.8938 |
| Qwen3.5-4B | `results/benchmark_qwen35_4b_no_retrans_logcost_hard2_k1024_5seed` | 320 | -0.244537 | `hybrid_heuristic` | -0.245694 | 4.5543 | 18.9674 | 0.84879 | +0.001157 | 0.7313 |

For Qwen3-0.6B, the log-cost and dense raw-drop calibration remove the pathological PPL tail, and hard-state fine-tuning raises the win/tie rate to `0.8938`. For Qwen3.5-4B, the final no-retransmission result uses the layer-wise MLP surrogate, hard-state fine-tuning, and a larger pure-policy inference pool (`k=1024`). The RL candidate pool still contains only learned-policy actions; strong heuristics are benchmark baselines and hard-state training teachers only.

The method tables below compare `autoreg_rl_pure` with the strongest heuristic baselines under both retransmission settings. The `r = 1` tables are the default retransmission-aware benchmark. The `r = 0` tables use log-cost hard-state reruns.

### r=1 Qwen3-0.6B Methods

| method | reward | feasible | latency | PPL_hat | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.571681 | 1.0000 | 2.6322 | 54.5724 | 0.21232 |
| hybrid_heuristic | -0.591107 | 1.0000 | 2.6323 | 56.0683 | 0.01839 |
| block_lns_strong | -0.591107 | 1.0000 | 2.6323 | 56.0683 | 0.09553 |
| block_beam_strong | -0.626139 | 1.0000 | 2.7118 | 58.1541 | 0.47908 |
| beam_search | -13.959931 | 1.0000 | 3.1348 | 1081.9976 | 0.07844 |

### r=1 Qwen3.5-4B Methods

| method | reward | feasible | latency | PPL_hat | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.164647 | 1.0000 | 4.4425 | 13.3604 | 0.11292 |
| hybrid_heuristic | -0.164903 | 1.0000 | 4.4400 | 13.3706 | 0.01815 |
| block_lns_strong | -0.164903 | 1.0000 | 4.4400 | 13.3706 | 0.08182 |
| block_beam_strong | -0.166586 | 1.0000 | 4.4711 | 13.3939 | 0.89049 |
| beam_search | -0.742398 | 1.0000 | 4.7914 | 30.9303 | 0.09164 |

### r=1 Gemma-4-E4B Methods

| method | reward | feasible | latency | PPL_hat | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.278169 | 1.0000 | 9.2066 | 10.7976 | 0.10848 |
| hybrid_heuristic | -0.278862 | 1.0000 | 9.2292 | 10.7980 | 0.02683 |
| block_lns_strong | -0.278862 | 1.0000 | 9.2292 | 10.7980 | 0.09493 |
| block_beam_strong | -0.281125 | 1.0000 | 9.3036 | 10.7988 | 0.91360 |
| beam_search | -0.304512 | 1.0000 | 10.0208 | 10.8491 | 0.19476 |

### r=0 Qwen3-0.6B Methods (log-cost, raw dense, hard states)

| method | reward | feasible | latency | PPL_hat | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.257262 | 1.0000 | 2.3714 | 32.4630 | 0.11442 |
| hybrid_heuristic | -0.259101 | 1.0000 | 2.3828 | 32.5285 | 0.01378 |
| block_lns_strong | -0.259101 | 1.0000 | 2.3828 | 32.5285 | 0.06987 |
| block_beam_strong | -0.264854 | 1.0000 | 2.4271 | 32.6427 | 0.43397 |
| beam_search | -0.291034 | 1.0000 | 2.6276 | 33.1968 | 0.08549 |
| simulated_annealing | -0.347934 | 1.0000 | 2.8706 | 36.2588 | 0.00723 |
| local_search | -0.348585 | 1.0000 | 2.8752 | 36.2784 | 0.01641 |

### r=0 Qwen3.5-4B Methods (log-cost, hard states, k=1024)

| method | reward | feasible | latency | PPL_hat | runtime_s |
|---|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.244537 | 1.0000 | 4.5543 | 18.9674 | 0.84879 |
| hybrid_heuristic | -0.245694 | 1.0000 | 4.5491 | 19.4734 | 0.02542 |
| block_lns_strong | -0.247842 | 1.0000 | 4.5387 | 19.9200 | 0.10289 |
| block_beam_strong | -0.252809 | 1.0000 | 4.5648 | 20.3846 | 3.65250 |
| beam_search | -0.422710 | 1.0000 | 4.9062 | 731.9199 | 0.46478 |
| simulated_annealing | -0.736104 | 1.0000 | 4.9292 | 666.5776 | 0.01175 |
| local_search | -0.739525 | 1.0000 | 4.8973 | 666.9373 | 0.03054 |

## Real LLM Validation

Real LLM validation recomputes paper-formula PPL for selected deployment actions. It is slower than simulator benchmarking and is used to validate the surrogate.

| model | artifact | rows | mean rel error | max rel error | RMSE log-ratio | Pearson | Spearman | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | `results/real_action_ppl_validation_layer_calibrated_retrained` | 64 competitive | 0.023323 | 0.216265 | 0.052984 | 0.996233 | 0.973764 | validated |
| Qwen3.5-4B | not available | 0 | - | - | - | - | - | pending |
| Gemma-4-E4B | not available | 0 | - | - | - | - | - | pending |

## Surrogate Benchmarks

Surrogate quality is reported separately for each model. These benchmarks are calibration-set fits built from the real-LLM profile directories. They are not held-out test benchmarks: each surrogate is fit on the sampled corruption curve stored in the real profile, then scored on those same sampled corruption points.

| model | surrogate artifact | clean PPL | fit type | R2 | RMSE log-ratio | mean rel error | max rel error | interpretation |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | `results/surrogate_benchmark_qwen3_0p6b` | 30.811979 | embedding exponential + layer gamma | 0.997363 | 0.019289 | 0.015285 | 0.030760 | strong |
| Qwen3-0.6B raw-dense | `results/surrogate_benchmark_qwen3_0p6b_raw_dense` | 30.811979 | embedding exponential + piecewise curve | 1.000000 | 0.000000 | 0.000000 | 0.000000 | dense raw-drop calibration; piecewise fit on 13 sampled points |
| Qwen3.5-4B | `results/surrogate_benchmark_qwen35_4b_v2` | 12.388740 | layer one-hot MLP | 0.999725 | 0.001998 | - | - | strongest surrogate fit |
| Gemma-4-E4B | `results/surrogate_benchmark_gemma4_e4b` | 10.744652 | empirical piecewise curve | 1.000000 | 0.000000 | 0.000000 | 0.000000 | fit R2 target met; exponential baseline R2 0.891476 |

Qwen3.5 also has a high-quality layer-wise analytic calibration: layer gamma sum `206.094048`, mean layer R2 `0.995787`, and layer RMSE log-ratio `0.006596`. Gemma's curve surrogate is fitted on 9 sampled embedding-drop points; it meets the R2 target on that curve, but it is still not a full layer-wise calibration. A compact cross-model surrogate report is stored at `results/surrogate_benchmark_multi_model_report.md`.

### How the surrogate benchmarks are built

All surrogate benchmarks use a real-LLM profile directory, then compare surrogate predictions against real PPL measured on corrupted forward passes. The table below shows the evaluation-text count and corruption grid used by each benchmark.

| model | real profile source | validation texts | corruption grid | trials per point | rows in the report | fit / evaluation |
|---|---|---:|---|---:|---:|---|
| Qwen3-0.6B | `results/qwen3_0p6b_real_profile` | 128 Wikitext-2 validation texts | 8 embedding-drop points from `0.0` to `0.1` | 6 | 8 curve points | exponential fit on `log(PPL/PPL_ref)`; report R2 and relative error on the sampled calibration points |
| Qwen3.5-4B | `results/qwen35_4b_real_profile_v2` | 64 Wikitext-2 validation texts | 31 boundary layers x 5 positive layer-drop points | 3 | 155 layer rows | layer-wise calibration plus layer-onehot MLP; report layer R2 and MLP R2 on the calibration rows |
| Gemma-4-E4B | `results/gemma4_e4b_real_profile` | 16 Wikitext-2 validation texts | 9 embedding-drop points from `0.0` to `0.1` | 3 | 9 curve points | linear baseline plus empirical piecewise curve over a scalar damage proxy; report both R2 values on the calibration grid |

For Qwen3-0.6B and Gemma-4-E4B, the benchmark uses embedding-drop corruption directly on the input embedding layer. For Qwen3.5-4B, the benchmark uses layer-wise hidden-state corruption and then trains the MLP surrogate on the resulting `layer_ppl_curve.json`. The reported `R2` values are calibration R2 values, not a separate held-out test-set score.

Qwen3-0.6B uses the exact drop grid `[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]` from `configs/real_llm.yaml`. The new no-retransmission raw-dense profile uses `[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1]` to better cover the larger raw PDP range. Gemma-4-E4B uses a denser empirical grid in `results/gemma4_e4b_real_profile/ppl_corruption_curve.json`; the piecewise fit reaches `R2 = 1.0` on those sampled points, but that is interpolation on the calibration curve, not an unseen generalization score.

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

![Gemma-4-E4B surrogate curve](results/surrogate_benchmark_gemma4_e4b/surrogate_ppl_fit.png)

## Reproduce

The tracked repository is sufficient to run the simulator, load the released
checkpoints, regenerate surrogate-fit reports, launch RL training, and reproduce
the simulator benchmark tables above. The benchmark directories now include
`benchmark_rows.csv`, so the reported aggregate metrics can be recomputed from
the tracked per-state rows.

Expected reproducibility scope:

| target | status | notes |
|---|---|---|
| Code import / compilation | reproducible | `python -m compileall -q src` should pass. |
| Surrogate benchmark from tracked profiles | reproducible | Regenerates the reported Qwen3-0.6B and Gemma curve-fit metrics. |
| Checkpoint benchmark rerun | reproducible | Use the commands below and compare with the tracked `benchmark_summary.csv` and `benchmark_margin.json`. |
| Full RL training | runnable, stochastic | Long run; final policy is not guaranteed to match the released checkpoint exactly. |
| Real LLM validation | hardware/model-cache dependent | Requires loading the requested Hugging Face model locally. |

Train Qwen3-0.6B:

```powershell
python -m src.train_autoreg_rl --config configs/qwen3_calibrated.yaml
```

Train Qwen3.5-4B:

```powershell
python -m src.train_autoreg_rl --config configs/qwen35_4b_calibrated.yaml
```

Train Gemma-4-E4B:

```powershell
python -m src.train_autoreg_rl --config configs/gemma4_base_calibrated.yaml
```

The config files contain the output directory, real-profile directory, teacher
warm-start settings, candidate mode, projection mode, and CUDA device. Training
is stochastic, so the exact checkpoint can vary across environments. The tracked
Qwen3.5-4B checkpoint is a hard-state fine-tune: it samples the losing states
from `results/benchmark_qwen35_4b_v2_teacher_big_5seed/benchmark_rows.csv` and
adds their strongest non-RL actions to replay, but benchmark-time inference is
still pure learned-policy candidate generation.

Benchmark a trained policy:

Qwen3-0.6B:

```powershell
python -m src.benchmark_real_profile --config configs/qwen3_calibrated.yaml
```

Expected summary: `autoreg_rl_pure` reward `-0.571681`, `PPL_hat` `54.5724`, win/tie `0.6813`.

Qwen3.5-4B:

```powershell
python -m src.benchmark_real_profile --config configs/qwen35_4b_calibrated.yaml
```

Expected summary: `autoreg_rl_pure` reward `-0.164647`, `PPL_hat` `13.3604`, win/tie `0.9250`.

Gemma-4-E4B:

```powershell
python -m src.benchmark_real_profile --config configs/gemma4_base_calibrated.yaml
```

Expected summary: `autoreg_rl_pure` reward `-0.278169`, `PPL_hat` `10.7976`, win/tie `0.7552`.

No-retransmission ablation (`r = 0`):

```powershell
# Generate the denser raw-drop real profile used by the Qwen3-0.6B no-retransmission rerun.
conda run -n LLM-UAV python -m src.real_llm_profile `
  --config configs/real_llm_qwen3_raw_dense.yaml `
  --fit-model piecewise

conda run -n LLM-UAV python -m src.train_autoreg_rl `
  --config configs/qwen3_no_retrans_logcost.yaml

conda run -n LLM-UAV python -m src.train_autoreg_rl `
  --config configs/qwen35_4b_no_retrans_logcost.yaml
```

The no-retransmission configs are hard-state fine-tunes. They use tracked compact hard-state specs in `hard_states_used.csv` and the released hard checkpoints for final benchmarking. The training commands above are runnable stochastic retraining commands; the table values are reproduced by benchmarking the tracked checkpoints. To regenerate the exact hard-state specs from scratch, first run a base log-cost benchmark, then point `ar_rl.hard_benchmark_rows` at that base `benchmark_rows.csv`. The Qwen3.5-4B final benchmark uses `k=1024` learned-policy candidates; this is slower than `k=256` but was needed to reliably beat the strongest heuristic on the 320-state benchmark.

Benchmark the no-retransmission checkpoints:

```powershell
conda run -n LLM-UAV python -m src.benchmark_real_profile `
  --config configs/qwen3_no_retrans_logcost.yaml

conda run -n LLM-UAV python -m src.benchmark_real_profile `
  --config configs/qwen35_4b_no_retrans_logcost.yaml
```

Expected no-retransmission summaries: Qwen3-0.6B log-cost/raw-dense hard-state `autoreg_rl_pure` reward `-0.257262`, `PPL_hat` `32.4630`, win/tie `0.8938`; Qwen3.5-4B log-cost hard-state `autoreg_rl_pure` reward `-0.244537`, `PPL_hat` `18.9674`, win/tie `0.7313`.

Run a surrogate benchmark:

```powershell
python -m src.benchmark_surrogate `
  --real-dir results/qwen3_0p6b_real_profile `
  --out results/surrogate_benchmark_qwen3_0p6b

python -m src.benchmark_surrogate `
  --real-dir results/qwen3_0p6b_real_profile_raw_dense `
  --out results/surrogate_benchmark_qwen3_0p6b_raw_dense `
  --fit piecewise `
  --write-surrogate
```

The surrogate benchmark is deterministic from the tracked profile files. For
Qwen3-0.6B, the original profile should reproduce an exponential-fit `R2`
around `0.997363`, and the raw-dense no-retrans profile should reproduce the
piecewise calibration report with `R2 = 1.0` on the sampled points. For
Gemma-4-E4B, the reported `R2 = 1.0` is interpolation on the calibration
curve, not held-out generalization.

## Repository Layout

```text
.
|-- configs/
|   |-- qwen3_calibrated.yaml
|   |-- qwen3_no_retrans_logcost.yaml
|   |-- qwen35_4b_calibrated.yaml
|   |-- qwen35_4b_no_retrans_logcost.yaml
|   |-- gemma4_base_calibrated.yaml
|   |-- real_llm.yaml
|   |-- real_llm_qwen3_raw_dense.yaml
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
|   `-- reports/
|       |-- make_autoreg_report.py
|       |-- make_k_sweep_report.py
|       |-- make_multi_model_surrogate_report.py
|       `-- make_visuals.py
|-- results/
|   |-- qwen3_0p6b_real_profile/
|   |-- qwen3_0p6b_real_profile_raw_dense/
|   |-- qwen35_4b_real_profile_v2/
|   |-- gemma4_e4b_real_profile/
|   |-- autoreg_rl_layer_calibrated_hard_k256/
|   |-- autoreg_rl_qwen35_4b_v2_teacher_big/
|   |-- autoreg_rl_gemma4_e4b_teacher/
|   |-- autoreg_rl_qwen3_0p6b_no_retrans_logcost_hard/
|   |-- autoreg_rl_qwen35_4b_no_retrans_logcost_hard2/
|   |-- benchmark_layer_calibrated_hard_k256_5seed/
|   |-- benchmark_qwen35_4b_v2_teacher_big_5seed/
|   |-- benchmark_gemma4_e4b_teacher/
|   |-- benchmark_qwen3_0p6b_no_retrans_logcost_hard_5seed/
|   |-- benchmark_qwen35_4b_no_retrans_logcost_hard2_k1024_5seed/
|   |-- real_action_ppl_validation_layer_calibrated_retrained/
|   |-- surrogate_benchmark_qwen3_0p6b/
|   |-- surrogate_benchmark_qwen3_0p6b_raw_dense/
|   |-- surrogate_benchmark_qwen35_4b_v2/
|   |-- surrogate_benchmark_gemma4_e4b/
|   |-- surrogate_benchmark_multi_model_report.md
|   |-- visuals_layer_calibrated_hard_k256/
|   `-- visuals_qwen35_teacher_big/
|-- requirements.txt
`-- README.md
```

Generated experiment outputs are ignored by default. The repository tracks only curated best checkpoints, profile files, benchmark summaries/reports/rows, selected train/eval logs for README figures, validation reports, surrogate summaries, and README figures. Teacher caches, stdout/stderr logs, smoke runs, and non-best checkpoints are intentionally not tracked.

## Setup

```powershell
conda create -n LLM-UAV python=3.11 -y
conda activate LLM-UAV
pip install numpy pandas pyyaml matplotlib transformers datasets
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
