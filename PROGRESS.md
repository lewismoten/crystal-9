# Crystal-9 verified progression

## Accepted INT4 layout

`mixed-int4-row-input-attention-q-v-out-k-output-bias-router-weight-input-bias-router-bias-expert-biases-output-bias`

- Token and positional tables: INT4 per row
- Attention Q/K/V/output projection weights: INT4 per row
- Attention output bias: INT4
- Attention input bias: INT4
- Router weight: INT4 per row
- Router bias: INT4
- All routed-expert matrix weights and biases: INT4 per row / INT4
- Output weight: INT4 per row
- Output bias: INT4
- Exact policy gate: **0 / 294,778** misses in both fake-QAT and materialized runtime

The accepted output-bias run uses learning rate `0.0001` and seed `20260928`, initialized from the verified expert-bias stage. It is not a full-model INT4 claim: LayerNorm parameters, packed storage, independent packed runtime, and invalid-input publication gates remain unfinished.

## Rejected router-weight trials

| Trial | Exact-policy misses |
|---|---:|
| Initial direct materialization | 11 |
| First 200-epoch QAT | 1 |
| 400-epoch continuation | 2 |
| 200 epochs, lower rate | 1 |
| 200 epochs, higher rate | 6 |

