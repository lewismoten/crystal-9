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

The accepted full-parameter run is a 300-epoch continuation from the one-miss group-of-2 checkpoint, using learning rate `0.00005` and seed `20260934`. The independent packed runtime is accepted as `crystal-9-packed-int4-v1`: it consumes signed INT4 nibbles and explicit scale tensors, and passed the same `0 / 294,778` legal-policy gate. Its artifact is `artifacts/crystal-9-int4-group2-packed-v1.pt` (`46,547` bytes; SHA-256 `10fb96a66aafb55b1841a0b90c3a2c2a8cfa0b24a67ac02da12434a2c722b9f9`). Integrity verification and invalid-history gates are accepted: malformed, repeated-square, oversized, and post-terminal histories return `!`.

A separate accepted scale-compressed deployment variant, `crystal-9-packed-int4-fp16-scales-v1`, preserves the same signed INT4 codes while storing every one of its 787 dequantization scales as FP16. Its independent runtime passed the complete `0 / 294,778` legal-policy gate. Artifact: `artifacts/crystal-9-int4-group2-packed-fp16-scales-v1.pt` (`46,299` bytes; SHA-256 `63eee663a143ee478308144da406873c72c05b6d5226dbb2f5e329dacb1392eb`). This is a representation-level compression candidate; the original FP32-scale artifact remains immutable and accepted.

## Accepted INT3 scope 1

`mixed-int3-suffix`

- INT3 scope: both matrix weights in each of the nine routed experts, plus `output.weight`, all rowwise INT3.
- Upstream tensors remain F32; this is an accepted staged scope, **not** a full-parameter INT3 model.
- A 300-epoch continuation from the one-miss 200-epoch checkpoint used seed `20260936` and learning rate `0.00005`.
- Exact policy gate: **0 / 294,778** misses in fake-QAT and separately materialized evaluation.
- The checkpoint's FP32 master has `181 / 294,778` misses; it is QAT state only and does not supersede the immutable F32 reference.
- A packed INT3 artifact/runtime has not yet been exported or accepted.

## Accepted INT3 scope 2

`mixed-int3-suffix-output-bias`

- Scope: accepted INT3 suffix plus `output.bias` as per-tensor INT3.
- Source: accepted INT3 suffix checkpoint; no retraining was required because direct materialization preserved the exact policy.
- Exact policy gate: **0 / 294,778** misses in both fake-QAT and separately materialized evaluation.
- Immutable report: `artifacts/int3-suffix-output-bias-accepted-report.json`.

## Rejected INT3 input-table trial

`mixed-int3-suffix-input`

- Scope: accepted INT3 suffix plus rowwise INT3 token and positional embedding tables.
- Direct quantization baseline from the accepted suffix: `13,267 / 294,778` misses.
- 500-epoch QAT with only the input tables trainable, seed `20260937`, learning rate `0.0001`: `11,987 / 294,778` misses in both fake-QAT and materialized evaluation.
- The FP32 masters regressed to `19,978 / 294,778` misses. This path is rejected and must not be exported or presented as an accepted INT3 model.

## Rejected router-weight trials

| Trial | Exact-policy misses |
|---|---:|
| Initial direct materialization | 11 |
| First 200-epoch QAT | 1 |
| 400-epoch continuation | 2 |
| 200 epochs, lower rate | 1 |
| 200 epochs, higher rate | 6 |

