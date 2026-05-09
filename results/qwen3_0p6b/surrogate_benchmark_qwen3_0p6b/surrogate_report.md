# Surrogate Benchmark

Real profile directory: `results/qwen3_0p6b/qwen3_0p6b_real_profile`

| metric | value |
|---|---:|
| PPL_ref | 30.811979 |
| fit | exponential |
| gamma | 10.898646 |
| layer gamma sum | 293.322997 |
| layer mean R2 | 0.982527 |
| exponential R2 log-ratio | 0.997363 |
| R2 log-ratio | 0.997363 |
| RMSE log-ratio | 0.019289 |
| MAE PPL | 0.797233 |
| Max abs PPL error | 1.585637 |
| Mean relative PPL error | 0.015285 |
| Max relative PPL error | 0.030760 |

| drop_rate | real PPL | surrogate PPL | rel error |
|---:|---:|---:|---:|
| 0.000 | 30.811979 +/- 0.000000 | 30.811979 | 0.000000 |
| 0.005 | 32.680680 +/- 0.810792 | 32.537614 | 0.004378 |
| 0.010 | 33.651341 +/- 0.339569 | 34.359893 | 0.021056 |
| 0.020 | 37.939772 +/- 1.942279 | 38.316341 | 0.009925 |
| 0.030 | 41.639366 +/- 2.802815 | 42.728362 | 0.026153 |
| 0.050 | 51.549363 +/- 4.648030 | 53.135000 | 0.030760 |
| 0.080 | 74.989651 +/- 4.230291 | 73.684704 | 0.017402 |
| 0.100 | 92.800956 +/- 11.556875 | 91.630863 | 0.012609 |
