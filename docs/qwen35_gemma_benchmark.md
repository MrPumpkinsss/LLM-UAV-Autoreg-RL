# Qwen3.5-4B and Gemma-4-E4B-it Benchmark

This benchmark was run on branch `qwen35-gemma-explore`. The original Qwen3-0.6B experiment is unchanged and preserved by tag `v0.6b-stable`.

## Deployment Smoke Tests

| model | result | notes |
|---|---|---|
| `Qwen/Qwen3.5-4B` | loads and runs a short forward pass | 8.68 GiB weights; after load, about 6.73 GiB VRAM remained |
| `google/gemma-4-E4B-it` | loads and runs a short forward pass | 14.89 GiB weights; Transformers used CPU offload for some parameters |

## Real LLM Profile

| model | clean PPL | forward latency | profile quality |
|---|---:|---:|---|
| `Qwen/Qwen3.5-4B` | 11.1778 embedding profile, 12.3887 v2 layer profile | 167.9 ms | usable |
| `google/gemma-4-E4B-it` | 32287.1 | 1624.3 ms | not reliable with the current PPL pipeline |

Gemma-4-E4B-it successfully deployed, but its current clean PPL is extremely high on the configured text-only PPL evaluation, and the embedding-level surrogate fit has negative R2. That means the current tokenizer/prompt/PPL pipeline is not a valid basis for a simulator benchmark. It should not be compared against Qwen results until the Gemma PPL evaluation is fixed.

## Qwen3.5-4B Surrogate Calibration

The first quick layer calibration was too noisy: it used `16` texts, `64` tokens, and one corruption trial, giving mean layer R2 `0.856720` and RMSE log-ratio `0.028426`. The v2 calibration below is the current recommended profile.

Layer-wise hidden-state corruption calibration v2:

| metric | value |
|---|---:|
| layers | 32 |
| clean PPL | 12.388740 |
| max length | 128 |
| sample count | 64 |
| drop rates | 0.0, 0.005, 0.01, 0.02, 0.03, 0.05 |
| corruption trials | 3 |
| fitted gamma sum | 206.094048 |
| fitted R2 | 0.995787 |
| RMSE log-ratio | 0.006596 |
| layer gamma mean | 6.648195 |
| layer gamma min | 4.467947 |
| layer gamma max | 10.431587 |

An optional MLP surrogate is also trained from the same v2 curve:

| metric | value |
|---|---:|
| rows | 155 |
| MLP R2 | 0.999725 |
| MLP RMSE log-ratio | 0.001998 |
| hidden dim | 64 |
| depth | 2 |

The main simulator benchmark still uses the linear layer-gamma surrogate. The MLP is kept as a fallback/diagnostic model; its inference path clamps the residual feature at the calibrated maximum and then applies the learned sensitivity to the actual residual, which prevents non-monotonic extrapolation beyond the calibrated drop range.

## Qwen3.5-4B RL Benchmark

Because the 4B model is much larger than Qwen3-0.6B, the UAV memory and energy ranges were scaled in `configs/qwen35_4b_calibrated.yaml` while keeping `N=5` UAVs. The RL policy was trained specifically for the 32-layer Qwen3.5-4B profile; the 0.6B checkpoint cannot be reused because observation and action dimensions differ.

Benchmark setting:

| item | value |
|---|---:|
| seeds | 101, 102, 103, 104, 105 |
| states per seed | 16 |
| total states | 80 |
| RL candidates | 256 |
| strong baseline beam width | 32 |
| anneal steps | 128 |

Results:

| method | reward mean | reward std | feasible | latency | PPL | runtime s |
|---|---:|---:|---:|---:|---:|---:|
| block_lns_strong | -0.148590 | 0.050451 | 1.0000 | 4.4540 | 12.8524 | 0.06087 |
| hybrid_heuristic | -0.148590 | 0.050451 | 1.0000 | 4.4540 | 12.8524 | 0.01368 |
| block_beam_strong | -0.149528 | 0.050791 | 1.0000 | 4.4621 | 12.8739 | 0.68943 |
| autoreg_rl_pure | -0.152260 | 0.053592 | 1.0000 | 4.5131 | 12.9111 | 0.04096 |
| beam_search | -0.230918 | 0.246015 | 1.0000 | 4.8283 | 15.0544 | 0.07091 |
| simulated_annealing | -1.791802 | 4.083456 | 1.0000 | 5.2076 | 63.0454 | 0.00671 |
| local_search | -1.792286 | 4.083321 | 1.0000 | 5.2111 | 63.0573 | 0.01592 |
| pdp_aware_greedy | -2.307821 | 4.561740 | 1.0000 | 5.1288 | 79.1008 | 0.00069 |
| random | -100.000000 | 0.000000 | 0.0000 | 183.0535 | 2.8090e18 | 0.02408 |
| latency_greedy | -15793.649937 | 109257.3 | 0.9750 | 9.8148 | 9.0238e6 | 0.00008 |
| block_balanced | -189260.669975 | 1583938.8 | 0.9750 | 17.4562 | 6.4090e6 | 0.00008 |

RL margin vs best non-RL baseline:

| metric | value |
|---|---:|
| mean margin | -0.003670 |
| min margin | -0.102670 |
| win/tie rate | 0.4500 |
| strict win rate | 0.1000 |

## Interpretation

Qwen3.5-4B is deployable on this desktop for profiling and simulator experiments. The v2 surrogate is much stronger than the first quick calibration, and v2 RL fine-tuning reduces the gap to the strongest heuristic from `-0.259829` mean margin to `-0.003670`. RL still does not beat the strongest block heuristics under the current 4B resource setting, so this benchmark is useful as a larger-model stress test, not as the main win-rate result.

Gemma-4-E4B-it can be loaded, but the current text-only PPL evaluation is not valid enough for RL benchmarking. The next step for Gemma would be to fix model-specific prompting/tokenization and verify a reasonable clean PPL before running layer-wise calibration.
