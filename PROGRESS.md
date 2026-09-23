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

The accepted full-parameter run is a 300-epoch continuation from the one-miss group-of-2 checkpoint, using learning rate `0.00005` and seed `20260934`. It establishes an all-parameter INT4 *layout*, not packed storage or a runnable deployment artifact. Packed storage, an independent packed runtime, and invalid-input publication gates remain unfinished.

## Rejected router-weight trials

| Trial | Exact-policy misses |
|---|---:|
| Initial direct materialization | 11 |
| First 200-epoch QAT | 1 |
| 400-epoch continuation | 2 |
| 200 epochs, lower rate | 1 |
| 200 epochs, higher rate | 6 |

