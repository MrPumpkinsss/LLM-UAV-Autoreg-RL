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
| `Qwen/Qwen3.5-4B` | 11.1778 embedding profile, 12.5082 layer profile | 167.9 ms | usable |
| `google/gemma-4-E4B-it` | 32287.1 | 1624.3 ms | not reliable with the current PPL pipeline |

Gemma-4-E4B-it successfully deployed, but its current clean PPL is extremely high on the configured text-only PPL evaluation, and the embedding-level surrogate fit has negative R2. That means the current tokenizer/prompt/PPL pipeline is not a valid basis for a simulator benchmark. It should not be compared against Qwen results until the Gemma PPL evaluation is fixed.

## Qwen3.5-4B Surrogate Calibration

Layer-wise hidden-state corruption calibration:

| metric | value |
|---|---:|
| layers | 32 |
| clean PPL | 12.508157 |
| drop rates | 0.0, 0.01, 0.03, 0.05 |
| corruption trials | 1 |
| fitted gamma sum | 196.068559 |
| fitted R2 | 0.856720 |
| RMSE log-ratio | 0.028426 |
| layer gamma mean | 6.324792 |
| layer gamma min | 3.874879 |
| layer gamma max | 11.154610 |

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
| block_lns_strong | -0.161487 | 0.076514 | 1.0000 | 4.6116 | 11.8245 | 0.05828 |
| hybrid_heuristic | -0.161487 | 0.076514 | 1.0000 | 4.6116 | 11.8245 | 0.01274 |
| block_beam_strong | -0.163243 | 0.078328 | 1.0000 | 4.6255 | 11.8619 | 0.47319 |
| beam_search | -0.230150 | 0.216815 | 1.0000 | 4.7512 | 13.6262 | 0.03778 |
| autoreg_rl_pure | -0.421316 | 0.761613 | 1.0000 | 5.1026 | 18.6736 | 0.04015 |
| simulated_annealing | -2.355267 | 11.46220 | 0.9875 | 7.5733 | 1.1828e6 | 0.00638 |
| local_search | -2.359120 | 11.46154 | 0.9875 | 7.5603 | 1.1828e6 | 0.01519 |
| pdp_aware_greedy | -50.672868 | 413.2303 | 1.0000 | 5.3058 | 1422.761 | 0.00040 |
| random | -100.000000 | 0.000000 | 0.0000 | 181.1778 | 1.6384e20 | 0.02289 |
| latency_greedy | -27453.464555 | 200845.7 | 0.9750 | 20.8246 | 1.8735e6 | 0.00007 |
| block_balanced | -566228.647247 | 4981037.0 | 0.9750 | 27.8794 | 1.5827e7 | 0.00008 |

RL margin vs best non-RL baseline:

| metric | value |
|---|---:|
| mean margin | -0.259829 |
| min margin | -5.055748 |
| win/tie rate | 0.1250 |
| strict win rate | 0.0000 |

## Interpretation

Qwen3.5-4B is deployable on this desktop for profiling and simulator experiments. The RL policy reaches 100% feasibility and is much better after hard-state training, but it does not beat the strongest block heuristics under the current 4B resource setting. This benchmark is therefore useful as a larger-model stress test, not as the main win-rate result.

Gemma-4-E4B-it can be loaded, but the current text-only PPL evaluation is not valid enough for RL benchmarking. The next step for Gemma would be to fix model-specific prompting/tokenization and verify a reasonable clean PPL before running layer-wise calibration.
