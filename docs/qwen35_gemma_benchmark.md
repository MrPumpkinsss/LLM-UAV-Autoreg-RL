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
| seeds | 61, 62, 63, 64, 65 |
| states per seed | 16 |
| total states | 80 |
| RL candidates | 256 |
| strong baseline beam width | 32 |
| anneal steps | 128 |

The recommended checkpoint is `results/autoreg_rl_qwen35_4b_v2_teacher_big/autoreg_policy_best.pt`. It warm-starts from the previous teacher checkpoint, uses `1000` teacher states, `1000` teacher replay updates, and `2000` RL episodes. A later hard-state replay attempt was evaluated but is not recommended because it reduced the final benchmark win/tie rate.

Results:

| method | reward mean | reward std | feasible | latency | PPL | runtime s |
|---|---:|---:|---:|---:|---:|---:|
| autoreg_rl_pure | -0.146974 | 0.047001 | 1.0000 | 4.4509 | 12.8052 | 0.05259 |
| hybrid_heuristic | -0.147090 | 0.045727 | 1.0000 | 4.4668 | 12.7940 | 0.01355 |
| block_lns_strong | -0.147090 | 0.045727 | 1.0000 | 4.4668 | 12.7940 | 0.06099 |
| block_beam_strong | -0.148220 | 0.047234 | 1.0000 | 4.4904 | 12.8071 | 0.66905 |
| simulated_annealing | -0.847906 | 2.140493 | 1.0000 | 5.1566 | 33.8587 | 0.00677 |
| local_search | -0.849439 | 2.140961 | 1.0000 | 5.1624 | 33.9007 | 0.01624 |
| beam_search | -3.568186 | 28.094510 | 1.0000 | 4.7848 | 118.4562 | 0.06955 |
| random | -100.000000 | 0.000000 | 0.0000 | 188.7475 | 1.7558e20 | 0.02350 |
| pdp_aware_greedy | -176.206820 | 1545.160373 | 1.0000 | 5.1121 | 5465.0899 | 0.00066 |
| latency_greedy | -1589.308559 | 6804.913171 | 0.9875 | 10.5849 | 73305.6639 | 0.00007 |
| block_balanced | -87503.764450 | 740761.087968 | 0.9375 | 19.5737 | 1.1719e7 | 0.00008 |

RL margin vs best non-RL baseline:

| metric | value |
|---|---:|
| mean margin | +0.000116 |
| min margin | -0.014858 |
| win/tie rate | 0.8250 |
| strict win rate | 0.1250 |

Hard-state replay check:

| variant | mean margin | min margin | win/tie |
|---|---:|---:|---:|
| teacher-big checkpoint | +0.000116 | -0.014858 | 0.8250 |
| teacher-big + hard replay | -0.000134 | -0.015341 | 0.7750 |

## Interpretation

Qwen3.5-4B is deployable on this desktop for profiling and simulator experiments. The v2 surrogate is much stronger than the first quick calibration. Expanded teacher-assisted v2 training uses strong baseline actions only as supervised replay during training; at benchmark time, `autoreg_rl_pure` still samples only from the learned policy. This improves the earlier teacher run from `51.25%` win/tie to `82.50%` win/tie against the best non-RL baseline on the 80-state benchmark.

Gemma-4-E4B-it can be loaded, but the current text-only PPL evaluation is not valid enough for RL benchmarking. The next step for Gemma would be to fix model-specific prompting/tokenization and verify a reasonable clean PPL before running layer-wise calibration.
