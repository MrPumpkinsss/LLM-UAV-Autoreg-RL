# Larger LLM Deployment Exploration

This note records isolated deployment tests on branch `qwen35-gemma-explore`.
The stable Qwen3-0.6B baseline is preserved by tag `v0.6b-stable`.

## Hardware

| item | value |
|---|---:|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM | 16.0 GiB usable by PyTorch |
| RAM | about 50 GB |
| CUDA visible | yes |

## Candidates

| model | HF access | weight size | smoke result | recommendation |
|---|---:|---:|---|---|
| `Qwen/Qwen3.5-9B` | yes | 17.98 GiB | loads with `device_map: auto`, short forward works, CPU offload used | not recommended for full PPL/layer calibration on this machine |
| `Qwen/Qwen3.5-4B` | yes | 8.68 GiB | loads on this machine, short forward works | recommended larger replacement if moving beyond 0.6B |
| `google/gemma-3-4b-pt` | gated | 8.01 GiB | blocked by Hugging Face license access | can retry after accepting Gemma license |
| `google/gemma-4-E4B-it` | unknown/gated likely | 14.89 GiB | not used as main fallback because it is much tighter than Qwen3.5-4B | only test after Qwen3.5-4B |

## Smoke Test Results

`Qwen/Qwen3.5-4B`:

```json
{
  "loaded": true,
  "loader": "AutoModelForCausalLM",
  "load_s": 78.14,
  "forward_s": 0.41,
  "output_shape": [1, 8, 248320],
  "cuda_total_gib": 15.92,
  "cuda_free_before_gib": 14.68,
  "cuda_free_after_gib": 6.73
}
```

`Qwen/Qwen3.5-9B`:

```json
{
  "loaded": true,
  "loader": "AutoModelForCausalLM",
  "load_s": 149.52,
  "forward_s": 2.15,
  "output_shape": [1, 8, 248320],
  "cuda_total_gib": 15.92,
  "cuda_free_before_gib": 14.68,
  "cuda_free_after_gib": 1.24,
  "note": "Transformers reported that some parameters were offloaded to CPU."
}
```

Gemma:

```text
google/gemma-3-4b-pt returned 403 GatedRepoError.
The tracked Gemma base path now uses `configs/real_llm_gemma4_base.yaml`.
```

## Code Management

- Keep the existing 0.6B files unchanged:
  - `configs/real_llm.yaml`
  - `configs/qwen3_calibrated.yaml`
  - `results/qwen3_0p6b_real_profile/`
  - `results/autoreg_rl_layer_calibrated_hard_k256/`
- Larger-model configs are separate:
  - `configs/real_llm_qwen35_9b.yaml`
  - `configs/real_llm_qwen35_4b.yaml`
  - `configs/real_llm_gemma4_base.yaml`
- Use `src.smoke_llm_deploy` before running full PPL calibration:

```powershell
conda run -n LLM-UAV python -m src.smoke_llm_deploy `
  --config configs/real_llm_qwen35_4b.yaml `
  --mode forward
```

## Next Step

For a larger real-LLM experiment on this desktop, use `Qwen/Qwen3.5-4B`.
Run real profile calibration with reduced `max_length`, `num_texts`, and `corruption_trials` first, then scale up only if memory and runtime are acceptable.
