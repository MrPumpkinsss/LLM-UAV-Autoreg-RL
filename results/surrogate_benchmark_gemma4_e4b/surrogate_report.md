# Surrogate Benchmark

Real profile directory: `results/gemma4_e4b_real_profile`

| metric | value |
|---|---:|
| PPL_ref | 10.744652 |
| gamma | 7.665657 |
| layer gamma sum | unavailable |
| layer mean R2 | unavailable |
| R2 log-ratio | 0.366344 |
| RMSE log-ratio | 0.101227 |
| MAE PPL | 0.891383 |
| Max abs PPL error | 2.192472 |
| Mean relative PPL error | 0.058760 |
| Max relative PPL error | 0.139512 |

| drop_rate | real PPL | surrogate PPL | rel error |
|---:|---:|---:|---:|
| 0.000 | 10.744652 +/- 0.000000 | 10.744652 | 0.000000 |
| 0.010 | 11.611999 +/- 0.109632 | 11.600691 | 0.000974 |
| 0.030 | 15.715281 +/- 3.300610 | 13.522809 | 0.139512 |
| 0.050 | 14.401648 +/- 0.044801 | 15.763403 | 0.094555 |
