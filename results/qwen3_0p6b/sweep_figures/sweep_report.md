# Parameter Sweep Figures

Config: `configs/qwen3_calibrated.yaml`
Policy: `results/qwen3_0p6b/autoreg_rl_layer_calibrated_hard_k256/autoreg_policy_best.pt`
Real profile: `results/qwen3_0p6b/qwen3_0p6b_real_profile`
UAVs: `5`
States per point: `24` (`91,92` x `12`)
Autoreg candidates: `256`

## Sweeps

### bandwidth_mhz

| value | RL reward | best non-RL reward | margin | win/tie | RL latency | RL PPL_hat | RL energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | -0.515930 | -0.519055 (block_lns_strong) | 0.003126 | 0.7083 | 4.6450 | 34.7738 | 1242.36 |
| 1 | -0.424451 | -0.427381 (block_lns_strong) | 0.002930 | 0.7083 | 3.7302 | 34.7738 | 998.43 |
| 1.5 | -0.332935 | -0.335824 (block_lns_strong) | 0.002889 | 0.7500 | 2.8150 | 34.7738 | 754.39 |
| 2 | -0.287124 | -0.289775 (block_lns_strong) | 0.002652 | 0.8750 | 2.3569 | 34.7736 | 632.24 |
| 3 | -0.241256 | -0.242832 (block_lns_strong) | 0.001576 | 0.8750 | 1.9067 | 34.7084 | 512.20 |
| 4 | -0.218241 | -0.219426 (block_lns_strong) | 0.001185 | 0.8333 | 1.6858 | 34.6375 | 453.48 |

![bandwidth_mhz reward](bandwidth_mhz_reward.png)

![bandwidth_mhz margin](bandwidth_mhz_margin.png)

![bandwidth_mhz latency](bandwidth_mhz_latency.png)

![bandwidth_mhz PPL](bandwidth_mhz_ppl.png)

![bandwidth_mhz energy](bandwidth_mhz_energy.png)

### energy_scale

| value | RL reward | best non-RL reward | margin | win/tie | RL latency | RL PPL_hat | RL energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | -0.287124 | -0.289775 (block_lns_strong) | 0.002652 | 0.8750 | 2.3569 | 34.7736 | 632.24 |
| 0.85 | -0.287124 | -0.289775 (block_lns_strong) | 0.002652 | 0.8750 | 2.3569 | 34.7736 | 632.24 |
| 1 | -0.287124 | -0.289775 (block_lns_strong) | 0.002652 | 0.8750 | 2.3569 | 34.7736 | 632.24 |
| 1.15 | -0.287184 | -0.289775 (block_lns_strong) | 0.002591 | 0.8333 | 2.3639 | 34.7246 | 634.10 |
| 1.3 | -0.287184 | -0.289775 (block_lns_strong) | 0.002591 | 0.8333 | 2.3639 | 34.7246 | 634.10 |

![energy_scale reward](energy_scale_reward.png)

![energy_scale margin](energy_scale_margin.png)

![energy_scale latency](energy_scale_latency.png)

![energy_scale PPL](energy_scale_ppl.png)

![energy_scale energy](energy_scale_energy.png)

### area_m

| value | RL reward | best non-RL reward | margin | win/tie | RL latency | RL PPL_hat | RL energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | -0.198318 | -0.199137 (block_lns_strong) | 0.000819 | 0.6250 | 1.9831 | 30.8129 | 532.49 |
| 750 | -0.219670 | -0.222161 (block_lns_strong) | 0.002491 | 0.7083 | 2.1601 | 31.0937 | 579.71 |
| 1000 | -0.287124 | -0.289775 (block_lns_strong) | 0.002652 | 0.8750 | 2.3569 | 34.7736 | 632.24 |
| 1250 | -0.613057 | -0.637741 (block_lns_strong) | 0.024683 | 0.8333 | 2.6138 | 57.9014 | 700.62 |
| 1500 | -1.413123 | -2.116826 (block_lns_strong) | 0.703703 | 0.7917 | 3.0390 | 116.2556 | 810.57 |

![area_m reward](area_m_reward.png)

![area_m margin](area_m_margin.png)

![area_m latency](area_m_latency.png)

![area_m PPL](area_m_ppl.png)

![area_m energy](area_m_energy.png)

### sequence_length

| value | RL reward | best non-RL reward | margin | win/tie | RL latency | RL PPL_hat | RL energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | -0.168848 | -0.170642 (block_lns_strong) | 0.001794 | 0.8333 | 1.1865 | 34.6785 | 318.23 |
| 96 | -0.228039 | -0.230209 (block_lns_strong) | 0.002169 | 0.8333 | 1.7746 | 34.7084 | 476.01 |
| 128 | -0.287124 | -0.289775 (block_lns_strong) | 0.002652 | 0.8750 | 2.3569 | 34.7736 | 632.24 |
| 160 | -0.346047 | -0.348999 (block_lns_strong) | 0.002952 | 0.8333 | 2.9462 | 34.7736 | 790.30 |
| 192 | -0.404970 | -0.408294 (block_lns_strong) | 0.003324 | 0.8333 | 3.5354 | 34.7736 | 948.35 |

![sequence_length reward](sequence_length_reward.png)

![sequence_length margin](sequence_length_margin.png)

![sequence_length latency](sequence_length_latency.png)

![sequence_length PPL](sequence_length_ppl.png)

![sequence_length energy](sequence_length_energy.png)

### snr_threshold

| value | RL reward | best non-RL reward | margin | win/tie | RL latency | RL PPL_hat | RL energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | -0.235270 | -0.238167 (block_lns_strong) | 0.002896 | 0.7500 | 2.3273 | 31.0079 | 624.24 |
| 8 | -0.254659 | -0.256118 (block_lns_strong) | 0.001459 | 0.8333 | 2.3427 | 32.3828 | 628.43 |
| 10 | -0.287124 | -0.289775 (block_lns_strong) | 0.002652 | 0.8750 | 2.3569 | 34.7736 | 632.24 |
| 12 | -0.344246 | -0.348828 (block_lns_strong) | 0.004582 | 0.8333 | 2.3910 | 38.9115 | 641.24 |
| 16 | -0.580920 | -0.603309 (block_lns_strong) | 0.022389 | 0.8333 | 2.4299 | 56.8430 | 651.59 |

![snr_threshold reward](snr_threshold_reward.png)

![snr_threshold margin](snr_threshold_margin.png)

![snr_threshold latency](snr_threshold_latency.png)

![snr_threshold PPL](snr_threshold_ppl.png)

![snr_threshold energy](snr_threshold_energy.png)

### latency_ref_s

| value | RL reward | best non-RL reward | margin | win/tie | RL latency | RL PPL_hat | RL energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | -0.522817 | -0.526842 (block_lns_strong) | 0.004025 | 0.8333 | 2.3569 | 34.7736 | 632.24 |
| 4.5 | -0.365688 | -0.368764 (block_lns_strong) | 0.003076 | 0.8333 | 2.3569 | 34.7736 | 632.24 |
| 6 | -0.287124 | -0.289775 (block_lns_strong) | 0.002652 | 0.8750 | 2.3569 | 34.7736 | 632.24 |
| 7.5 | -0.239870 | -0.242122 (block_lns_strong) | 0.002252 | 0.8333 | 2.3661 | 34.7084 | 634.68 |
| 9 | -0.208322 | -0.210353 (block_lns_strong) | 0.002031 | 0.8333 | 2.3661 | 34.7084 | 634.68 |

![latency_ref_s reward](latency_ref_s_reward.png)

![latency_ref_s margin](latency_ref_s_margin.png)

![latency_ref_s latency](latency_ref_s_latency.png)

![latency_ref_s PPL](latency_ref_s_ppl.png)

![latency_ref_s energy](latency_ref_s_energy.png)
