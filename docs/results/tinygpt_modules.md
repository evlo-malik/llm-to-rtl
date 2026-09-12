| matrix | K x N | kind | cycles/vector | MACs/cycle | result |
|---|---|---|---|---|---|
| layer0_qkv | 64 x 192 | rom | 4991 | 2.5 | pass |
| layer0_proj | 64 x 64 | rom | 1663 | 2.5 | pass |
| layer0_ffwd1 | 64 x 256 | rom | 6655 | 2.5 | pass |
| layer0_ffwd2 | 256 x 64 | rom | 6463 | 2.5 | pass |
| layer1_qkv | 64 x 192 | rom | 4991 | 2.5 | pass |
| layer1_proj | 64 x 64 | rom | 1663 | 2.5 | pass |
| layer1_ffwd1 | 64 x 256 | rom | 6655 | 2.5 | pass |
| layer1_ffwd2 | 256 x 64 | rom | 6463 | 2.5 | pass |
| lm_head | 64 x 65 | rom | 1864 | 2.2 | pass |
