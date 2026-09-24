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

## Rejected INT3 router-bias trials

`mixed-int3-suffix-output-bias-router-bias`

- Scope: accepted INT3 suffix plus `output.bias`, with `router.bias` added as per-tensor INT3. All other tensors were frozen and recorded by SHA-256 in the per-run provenance.
- Direct materialization baseline: `6 / 294,778` misses in both fake-QAT and materialized evaluation.
- Isolated 200-epoch QAT, learning rate `0.0001`: seeds `20260938`, `20260939`, and `20260940` each reached `1 / 294,778` misses in both paths; seed `20260941` regressed to `10 / 294,778`.
- Every candidate is rejected: parity matched, but no candidate achieved the required `0 / 294,778`. Preserve these artifacts as evidence; do not export them as accepted INT3 model state or keep repeating the same recipe.
- Next work must be a separately preflighted strategy change or controlled continuation from a one-miss candidate, with exactly one changed variable.

## Rejected INT3 router-weight rowwise candidate

`mixed-int3-suffix-output-bias-router-weight`

- Scope: accepted INT3 suffix plus `output.bias`, with `router.weight` added as rowwise INT3.
- Direct materialization: `142 / 294,778` misses in both fake-QAT and materialized paths.
- Isolated 200-epoch QAT, seed `20260942`, learning rate `0.0001`, reached `43 / 294,778` misses in both paths. The frozen predecessor inventory is SHA-256 asserted in its immutable report.
- This is a material improvement over direct quantization but is not near the `0 / 294,778` gate; it is rejected rather than extended.

## Active INT3 router-weight groupwise candidate

`mixed-int3-suffix-output-bias-router-weight-group4`

- Scope: accepted INT3 suffix plus `output.bias`, with `router.weight` as INT3 in independent contiguous four-value groups per row. This changes only router-weight quantization granularity from the rejected rowwise candidate.
- Parity test: `tests/test_mixed_int3_suffix_output_bias_router_weight_group4_parity.py` was red before implementation and passes; isolated-scope provenance test also passes.
- Exhaustive direct materialization from the accepted suffix checkpoint: `13 / 294,778` misses in fake-QAT and `13 / 294,778` materialized. The immutable preflight report is `artifacts/int3-suffix-output-bias-router-weight-group4-direct-materialization-report.json`; QAT is therefore required.
- Active recipe: 200 epochs, seed `20260943`, learning rate `0.0001`, with only `router.weight` trainable and all predecessor tensors SHA-256 asserted frozen.

## Accepted INT3 scope 3

`mixed-int3-suffix-output-bias-router-weight-group4`

- Scope: accepted INT3 suffix and `output.bias`, plus `router.weight` in independent contiguous four-value INT3 groups per row. Upstream tensors, including `router.bias`, remain F32; this remains a staged scope, not a full-parameter INT3 model.
- Source: accepted `mixed-int3-suffix` checkpoint `artifacts/int3-suffix-qat-300-continuation-seed20260936-lr5e-5/artifacts-qat-mixed-int3-suffix.pt`.
- Direct materialization missed `13 / 294,778`; isolated QAT trained only `router.weight` for 200 epochs at seed `20260943`, learning rate `0.0001`. SHA-256 assertions cover every frozen predecessor tensor.
- Exact policy gate: **0 / 294,778** misses in fake-QAT and separately materialized evaluation.
- Immutable report and checkpoint: `artifacts/int3-suffix-output-bias-router-weight-group4-qat-200-seed20260943-lr1e-4/report.json` and `artifacts/int3-suffix-output-bias-router-weight-group4-qat-200-seed20260943-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4.pt`.
- Next ordered work: separately preflight router-bias INT3 on this accepted groupwise-router-weight predecessor; do not infer acceptance from the earlier router-bias-only trials.

## Accepted INT3 scope 4

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias`

- Scope: accepted INT3 suffix and `output.bias`, plus `router.weight` in independent contiguous four-value INT3 groups per row and `router.bias` as per-tensor INT3. Upstream tensors remain F32; this remains a staged scope, not a full-parameter INT3 model.
- Source: accepted groupwise-router-weight checkpoint `artifacts/int3-suffix-output-bias-router-weight-group4-qat-200-seed20260943-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4.pt`.
- Direct materialization missed `1 / 294,778`; isolated QAT trained only `router.bias` for 200 epochs at seed `20260944`, learning rate `0.0001`. SHA-256 assertions cover every frozen predecessor tensor.
- Exact policy gate: **0 / 294,778** misses in fake-QAT and separately materialized evaluation.
- Immutable report and checkpoint: `artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-qat-200-seed20260944-lr1e-4/report.json` and `artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-qat-200-seed20260944-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4-router-bias.pt`.
- Next ordered work: preflight isolated INT3 routed-expert biases on this accepted predecessor. Do not create a packed INT3 artifact: input/attention/norm scopes remain F32 and lack independent runtime/integrity gates.

## Accepted INT3 scope 5

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases`

- Scope: accepted INT3 suffix and `output.bias`, plus `router.weight` in independent contiguous four-value INT3 groups per row, `router.bias` as per-tensor INT3, and both bias vectors in every routed expert as per-tensor INT3. Input, attention, and norm tensors remain F32; this is a staged scope, not a full-parameter INT3 model.
- Source: accepted groupwise router-weight and router-bias checkpoint `artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-qat-200-seed20260944-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4-router-bias.pt`.
- Direct materialization missed `7 / 294,778`; isolated QAT trained only `experts.*.0.bias` and `experts.*.2.bias` for 200 epochs, seed `20260945`, learning rate `0.0001`. SHA-256 assertions cover every frozen predecessor tensor.
- Exact policy gate: **0 / 294,778** misses in fake-QAT and separately materialized evaluation.
- Immutable report and checkpoint: `artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-qat-200-seed20260945-lr1e-4/report.json` (SHA-256 `a3ea618f8c967191a35167470b586645f88832b16c31cd2f090c4cf5f517b9b3`) and `artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-qat-200-seed20260945-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases.pt` (SHA-256 `dc3e77ca17224775b24d61f616c351165ef90f31ee83809889ab45fc30ac06f1`).
- Next ordered work: preflight isolated INT3 output bias and output weight together is not authorized because `output.bias` and `output.weight` are already accepted in the suffix. The next unresolved ordered component is the input embedding tables; retain the prior rejected rowwise candidate and establish a distinct parity-tested granularity strategy from this stronger predecessor. No packed INT3 artifact may be created while input, attention, and norm scopes remain F32 and independent runtime/integrity gates are absent.

