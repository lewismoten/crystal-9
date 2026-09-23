# Crystal-9 verified progression

## Accepted INT4 layout

`mixed-int4-row-input-attention-q-v-out-k-output-bias-router-weight`

- Token and positional tables: INT4 per row
- Attention Q/K/V/output projection weights: INT4 per row
- Attention output bias: INT4
- Attention input bias: INT4
- Router weight: INT4 per row
- All routed-expert matrix weights and output weight: INT4 per row
- Exact policy gate: **0 / 294,778** misses in both fake-QAT and materialized runtime

The accepted input-bias run uses learning rate `0.0001` and seed `20260925`, initialized from the verified router-weight stage. It is not a full-model INT4 claim: router bias, expert biases, output bias, LayerNorm parameters, packed storage, independent packed runtime, and invalid-input publication gates remain unfinished.

## Rejected router-weight trials

| Trial | Exact-policy misses |
|---|---:|
| Initial direct materialization | 11 |
| First 200-epoch QAT | 1 |
| 400-epoch continuation | 2 |
| 200 epochs, lower rate | 1 |
| 200 epochs, higher rate | 6 |

