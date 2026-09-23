# Crystal-9 verified progression

## Accepted INT4 layout

`mixed-int4-full-parameters-norm-weight-group2`

- Token and positional tables: INT4 per row
- Attention Q/K/V/output projection weights: INT4 per row
- Attention output bias: INT4
- Attention input bias: INT4
- Router weight: INT4 per row
- Router bias: INT4
- All routed-expert matrix weights and biases: INT4 per row / INT4
- Output weight: INT4 per row
- Output bias: INT4
- LayerNorm bias: INT4 per tensor
- LayerNorm weight: INT4 in 16 contiguous groups of 2, each with an explicit group scale
- Exact policy gate: **0 / 294,778** misses in both fake-QAT and materialized runtime

The accepted full-parameter run is a 300-epoch continuation from the one-miss group-of-2 checkpoint, using learning rate `0.00005` and seed `20260934`. The independent packed runtime is accepted as `crystal-9-packed-int4-v1`: it consumes signed INT4 nibbles and explicit scale tensors, and passed the same `0 / 294,778` legal-policy gate. Its artifact is `artifacts/crystal-9-int4-group2-packed-v1.pt` (`46,419` bytes; SHA-256 `f7daae2c717641528ed3ca1d311fc52b1402bb8ea75d9f4588b3c32a4484135a`). Invalid-input publication gates remain unfinished.

## Rejected router-weight trials

| Trial | Exact-policy misses |
|---|---:|
| Initial direct materialization | 11 |
| First 200-epoch QAT | 1 |
| 400-epoch continuation | 2 |
| 200 epochs, lower rate | 1 |
| 200 epochs, higher rate | 6 |

