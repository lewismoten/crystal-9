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

## Active INT3 groupwise-input candidate

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-input-group4`

- Scope: accepted INT3 scope 5 plus `embedding.weight` and `position.weight` as independent contiguous four-value INT3 groups per row. This is a distinct granularity strategy from the rejected rowwise-input trial, sourced from the stronger accepted expert-bias predecessor.
- Source: `artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-qat-200-seed20260945-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases.pt`.
- Parity test: `tests/test_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_input_group4_parity.py` was red before implementation and now passes. The isolated-scope runner test verifies only the two input tables are trainable and records SHA-256 values for every frozen predecessor tensor.
- Exhaustive direct-materialization preflight: `2,985 / 294,778` misses in fake-QAT and `2,985 / 294,778` materialized (`artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-input-group4-direct-materialization-report.json`). QAT is required.
- Active recipe: isolated 200-epoch QAT, seed `20260946`, learning rate `0.0001`, batch size `1024`; only the two named input tables are trainable. Acceptance remains exactly `0 / 294,778` in both paths.

## Rejected INT3 groupwise-input candidate

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-input-group4`

- Direct materialization missed `2,985 / 294,778`; isolated 200-epoch QAT trained only `embedding.weight` and `position.weight` at seed `20260946`, learning rate `0.0001`, reducing matching fake-QAT/materialized misses to `835 / 294,778`.
- This is a material improvement but not near the exact gate. The immutable checkpoint and report remain rejected evidence; it will not be extended or exported.

## Rejected INT3 two-value-groupwise-input candidate

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-input-group2`

- Direct materialization missed `862 / 294,778`; isolated 200-epoch QAT trained only `embedding.weight` and `position.weight` at seed `20260947`, learning rate `0.0001`, reducing matching fake-QAT/materialized misses to `121 / 294,778`.
- This is a material improvement over group4 but is not near the exact gate. The immutable checkpoint and report remain rejected evidence; this combined-table recipe will not be extended.

## Rejected INT3 position-table direct-materialization preflight

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-group2`

- Scope: accepted INT3 scope 5 plus only `position.weight` as independent contiguous two-value INT3 groups per row; `embedding.weight` remains F32.
- Direct materialization from the accepted scope-5 checkpoint produced matching fake-QAT/materialized totals of `114 / 294,778` (`artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-group2-direct-materialization-report.json`). It is rejected as direct materialization and requires a distinct isolated QAT attempt.

## Rejected INT3 position-table group2 QAT candidate

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-group2`

- Strategy change: isolate only `position.weight` with two-value INT3 groups, rather than extending the rejected combined-table candidate. `embedding.weight` and all accepted predecessor tensors are SHA-256 asserted frozen.
- Parity test `tests/test_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group2_parity.py` was red before implementation and passes. The isolated runner test verifies `position.weight` is the sole trainable tensor.
- Initial isolated QAT: 200 epochs, seed `20260948`, learning rate `0.0001`, batch size `1024`, reached matching fake-QAT/materialized `4 / 294,778` misses.
- One controlled continuation from that checkpoint changed only learning rate to `0.00005` for 100 epochs (same seed); it regressed to matching `6 / 294,778` misses. Both immutable artifacts are rejected. Do not extend this recipe again.

## Rejected INT3 position-table rowwise direct-materialization candidate

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-rowwise`

- Distinct granularity strategy: accepted scope 5 plus only `position.weight` in rowwise INT3, with `embedding.weight` and every predecessor tensor retained F32/accepted layout.
- The new fake-QAT/materialized parity test was observed red before implementation and now passes: `tests/test_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_rowwise_parity.py`.
- Exhaustive direct materialization from accepted scope 5 produced matching `2,511 / 294,778` misses (`artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-rowwise-direct-materialization-report.json`), worse than the group2 direct preflight (`114 / 294,778`). This direct layout is rejected; no QAT has been started.

## Accepted INT3 scope 6

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-group1`

- Scope: accepted INT3 scope 5 plus only `position.weight`, represented by independent scalar INT3 groups; `embedding.weight`, attention, and norm tensors remain F32.
- No QAT was run. The parity-tested direct-materialization gate passed with matching **0 / 294,778** fake-QAT and materialized-policy misses from the immutable scope-5 checkpoint.
- Immutable report: `artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-group1-direct-materialization-report.json`.
- This is a legitimate scalar-group representation, but it carries one scale per position scalar and is therefore materially less storage-efficient than group-2. It does not constitute a packed INT3 artifact or a full-parameter INT3 model.

## INT3 input stage status

- Accepted staged scope is now scope 6: expert matrices/biases, `output.weight`, `output.bias`, `router.weight` (four-value groups), `router.bias`, and scalar-group `position.weight`; `embedding.weight`, attention, and norm tensors remain F32.

## Accepted INT3 scope 7

`mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-group1-embedding-group1`

- Scope: accepted INT3 scope 6 plus only `embedding.weight`, represented by independent scalar INT3 groups. Attention and norm tensors remain F32.
- The new embedding parity test was observed red for the absent layout methods, then passed after their minimal implementation: `tests/test_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group1_embedding_group1_parity.py`.
- No QAT was run. Direct materialization from the immutable scope-5 checkpoint produced matching **0 / 294,778** fake-QAT and materialized-policy misses.
- Immutable report: `artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-position-group1-embedding-group1-direct-materialization-report.json`.
- Scalar groups preserve this policy exactly but require one scale per input scalar; this is not a storage-efficient packed INT3 artifact or a full-parameter INT3 model.

## INT3 input stage status

- Accepted staged scope is now scope 7: expert matrices/biases, `output.weight`, `output.bias`, `router.weight` (four-value groups), `router.bias`, and scalar-group position and token tables; attention and norm tensors remain F32.
- Both scalar-group input-table layouts passed direct materialization without QAT.

## Accepted INT3 scope 8

`mixed-int3-scalar-input-attention-q`

- Scope: accepted INT3 scope 7 plus only the Q rows (`attention.in_proj_weight[:32]`) in rowwise INT3. K/V rows, attention output projection, both attention biases, and norm tensors remain F32.
- The Q-layout fake-QAT/materialized parity test was red before implementation and now passes: `tests/test_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group1_embedding_group1_attention_q_parity.py`.
- Direct materialization from the immutable scope-5 checkpoint missed matching `16 / 294,778` legal policies, so QAT was required.
- Initial isolated QAT trained only Q rows for 200 epochs, seed `20260949`, learning rate `0.0001`, and reached matching `2 / 294,778` misses. It remains immutable rejected evidence.
- One controlled continuation changed only the learning rate to `0.00005` for 100 epochs from that two-miss checkpoint, preserving every frozen tensor by SHA-256. It reached matching **0 / 294,778** fake-QAT and materialized-policy misses.
- Immutable reports: `artifacts/int3-scalar-input-attention-q-qat-200-seed20260949-lr1e-4/report.json` and `artifacts/int3-scalar-input-attention-q-qat-continue-100-seed20260949-lr5e-5/report.json`.
- This is still a staged mixed-precision model, not a packed INT3 artifact or full-parameter INT3 model.

## INT3 stage status

- Accepted staged scope is now scope 8: experts, output, router, scalar-group token and position tables, and attention Q rows are INT3 under their recorded layouts. Attention K/V/output, attention biases, and norm tensors remain F32.
- No training process is active: the controlled Q continuation reached the exact gate. The next unresolved component is attention K, which needs a new parity-tested direct-materialization preflight.

## Rejected INT3 attention K rowwise candidate

`mixed-int3-scalar-input-attention-q-k`

- Scope: accepted INT3 scope 8 plus only the K rows (`attention.in_proj_weight[32:64]`) in rowwise INT3. Q rows remained at their accepted INT3 state; V rows and every other tensor were frozen and asserted.
- Exhaustive direct materialization from the accepted Q checkpoint produced matching `2,396 / 294,778` fake-QAT/materialized misses.
- Isolated 200-epoch K-only QAT, seed `20260950`, learning rate `0.0001`, regressed to matching `2,950 / 294,778` misses.
- This is rejected evidence, not an accepted stage. Do not continue this rowwise recipe; the next K attempt must change quantization granularity.

## Active INT3 attention K four-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group4`

- Scope: accepted INT3 scope 8 plus only K rows (`attention.in_proj_weight[32:64]`) in contiguous four-value INT3 groups per row. Q remains in its accepted rowwise INT3 state; V and every other predecessor tensor are frozen.
- The new fake-QAT/materialized parity test was red for the absent Q+K-group4 runner, then passes after its minimal implementation: `tests/test_mixed_int3_attention_q_k_group4_parity.py`. The isolated-runner test asserts exact Q/V and all non-projection predecessor tensor identity; its optimizer uses zero weight decay so gradient-masked Q/V slices cannot drift.
- Direct materialization from the accepted Q checkpoint produced matching `614 / 294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group4-direct-materialization/report.json`). The group layout has 1,024 FP32 scales for K and is a staged representation, not a packed INT3 release.
- Active recipe: isolated 200-epoch QAT, seed `20260951`, learning rate `0.0001`, batch size `1024`, with only K rows trainable. Acceptance remains matching exactly `0 / 294,778` misses.

## Rejected INT3 attention K four-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group4`

- The isolated K-only QAT completed at seed `20260951`, learning rate `0.0001`, 200 epochs with matching `72 / 294,778` fake-QAT/materialized-policy misses. This improves on its `614`-miss direct baseline but is not near the exact gate and is immutable rejected evidence; do not continue it.
- Provenance and frozen Q/V/predecessor SHA-256 inventory: `artifacts/int3-scalar-input-attention-q-k-group4-qat-200-seed20260951-lr1e-4/report.json`.

## Active INT3 attention K two-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2`

- Strategy change: only K granularity changes from rejected group4 to independent contiguous two-value INT3 groups per row; accepted Q rows remain rowwise INT3 and V/every other predecessor tensor remain frozen.
- The new fake-QAT/materialized parity test was observed red before implementation and is now green: `tests/test_mixed_int3_attention_q_k_group2_parity.py`. The isolated runner test asserts Q/V and all non-projection predecessor tensors remain exactly unchanged; AdamW uses zero weight decay while Q/V gradients are masked.
- Exhaustive direct materialization from accepted scope 8 produced matching `28 / 294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-direct-materialization/report.json`), so QAT is required. K has 512 FP32 scales in this group-2 staged representation; it is not a packed INT3 release.
- Isolated 200-epoch QAT, seed `20260952`, learning rate `0.0001`, batch size `1024`, trained only K rows with zero optimizer weight decay and exact SHA-256 assertions for Q, V, and every non-projection predecessor tensor.
- Exact policy gate: **0 / 294,778** misses in fake-QAT and separately materialized evaluation. Immutable report/checkpoint: `artifacts/int3-scalar-input-attention-q-k-group2-qat-200-seed20260952-lr1e-4/report.json` and `artifacts/int3-scalar-input-attention-q-k-group2-qat-200-seed20260952-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2.pt`.

## Active INT3 attention V four-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group4`

- Scope: accepted INT3 K group-2 predecessor plus only V rows (`attention.in_proj_weight[64:96]`) in contiguous four-value INT3 groups. Q remains rowwise INT3 and K remains group-2 INT3; all other tensors are frozen.
- New fake-QAT/materialized parity test: `tests/test_mixed_int3_attention_q_k_group2_v_group4_parity.py` was red for the absent layout and passes after implementation.
- Exhaustive direct materialization from the accepted K checkpoint produced matching `63 / 294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-v-group4-direct-materialization/report.json`), so QAT is required. V has 1,024 FP32 scales and is a staged representation, not a packed INT3 release.
- Active recipe: isolated 200 epochs, seed `20260953`, learning rate `0.0001`, batch size `1024`; only V is trainable, while Q/K slices and all predecessor tensors are SHA-256 asserted frozen.

## Active INT3 attention V four-value-group controlled continuation

`mixed-int3-scalar-input-attention-q-k-group2-v-group4`

- The initial isolated V-only candidate completed at matching `7 / 294,778` fake-QAT/materialized-policy misses. This is near the exact gate and immutable evidence, not accepted.
- Decision: **continue** once from its best checkpoint, changing only learning rate from `0.0001` to `0.00005`; same seed `20260953`, 100 epochs, batch size `1024`, and only V rows trainable. Q/K slices and all non-projection predecessor tensors remain SHA-256 asserted frozen.
- Running artifact directory: `artifacts/int3-scalar-input-attention-q-k-group2-v-group4-qat-continue-100-seed20260953-lr5e-5/`. Acceptance remains exactly `0 / 294,778` in both paths. A regression or nonzero total rejects this recipe; the next action will be a parity-tested V granularity change.

## Rejected INT3 attention V four-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group4`

- Initial isolated QAT reached matching `7 / 294,778` misses; its one controlled continuation changed only learning rate to `0.00005` for 100 epochs and regressed to matching `14 / 294,778` misses.
- Both group-4 V QAT artifacts are immutable rejected evidence. Do not extend this recipe again.

## Active INT3 attention V two-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group2`

- Strategy change: only V quantization granularity changes from rejected group-4 V to independent contiguous two-value INT3 groups per row. Accepted Q remains rowwise INT3, accepted K remains group-2 INT3, and all other predecessor tensors remain frozen.
- The fake-QAT/materialized parity test was observed red for absent group-2 V methods and passes after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group4_parity.py`. The isolated runner test asserts exact Q/K and all non-projection predecessor tensor identity; AdamW uses zero weight decay while Q/K gradients are masked.
- Exhaustive direct materialization from the accepted K checkpoint produced matching `10 / 294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-v-group2-direct-materialization/report.json`), so QAT is required. V has 512 FP32 scales and is a staged representation, not a packed INT3 release.
- Active recipe: isolated 200 epochs, seed `20260954`, learning rate `0.0001`, batch size `1024`; only V is trainable, while Q/K slices and all predecessor tensors are SHA-256 asserted frozen. Acceptance remains exactly `0 / 294,778` in both paths.

## Accepted INT3 scope 9

`mixed-int3-scalar-input-attention-q-k-group2-v-group2`

- Scope: accepted INT3 scope 8 plus V rows (`attention.in_proj_weight[64:96]`) in independent contiguous two-value INT3 groups. Q remains rowwise INT3 and K remains group-2 INT3; attention output projection/bias and norm tensors remain F32.
- Direct materialization missed `10 / 294,778`; isolated V-only QAT trained 200 epochs at seed `20260954`, learning rate `0.0001`, batch size `1024`, with Q/K slices and every non-projection predecessor tensor SHA-256 asserted frozen.
- Exact policy gate: **0 / 294,778** misses in fake-QAT and separately materialized evaluation. Immutable report/checkpoint: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/report.json` and `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt`.
- V uses 512 FP32 scales (group size 2), so this remains a staged representation, not a packed INT3 release.

## Active INT3 attention output-projection rowwise candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out`

- Scope: accepted INT3 scope 9 plus only `attention.out_proj.weight` in rowwise INT3. Q/K/V accepted layouts remain frozen, as do both attention biases and all other predecessor tensors.
- New fake-QAT/materialized parity test was observed red before implementation and now passes: `tests/test_mixed_int3_attention_q_k_group2_v_group4_parity.py`. The isolated runner test asserts only the attention output-projection weight can change.
- Exhaustive direct materialization from accepted scope 9 missed matching `2,086 / 294,778` fake-QAT/materialized legal policies (`artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-direct-materialization/report.json`); QAT is required.
- Initial isolated QAT: 200 epochs, seed `20260955`, learning rate `0.0001`, batch size `1024`, trained only `attention.out_proj.weight` with zero optimizer weight decay and SHA-256 inventory for every frozen tensor. It completed at matching `275 / 294,778` fake-QAT/materialized misses and is immutable rejected evidence, not a candidate to extend.

## Active INT3 attention output-projection four-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group4`

- Strategy change: only the output-projection quantization granularity changes from the rejected rowwise recipe to independent contiguous four-value INT3 groups; Q/K/V and every other predecessor tensor remain frozen.
- The new fake-QAT/materialized parity test was observed red before implementation and is green: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py`. The isolated-runner test asserts every tensor other than `attention.out_proj.weight` retains exact identity.
- Exhaustive direct materialization from accepted scope 9 produced matching `145 / 294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group4-direct-materialization/report.json`), so QAT is required. This layout has 256 FP32 scales and is a staged representation, not a packed INT3 release.
- Initial isolated QAT: 200 epochs, seed `20260956`, learning rate `0.0001`, batch size `1024`, trained only `attention.out_proj.weight` with zero optimizer weight decay and SHA-256 assertions for every frozen predecessor tensor. It reduced matching fake-QAT/materialized misses from `145` to `7 / 294,778`; this is immutable evidence, not acceptance.
- Decision: **change strategy**. The one allowed group-4 continuation regressed from matching `7` to `12 / 294,778` misses, so both group-4 QAT artifacts are immutable rejected evidence; do not extend them again.

## Active INT3 attention output-projection two-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group2`

- Strategy change: only `attention.out_proj.weight` granularity changes from rejected group-4 to independent contiguous two-value INT3 groups. Accepted Q/K/V and every other predecessor tensor remain frozen.
- The parity test was observed red for the absent group-2 output layout, then green after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py`. The isolated runner test asserts every tensor other than `attention.out_proj.weight` retains exact identity.
- Exhaustive direct materialization from accepted scope 9 produced matching `32 / 294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group2-direct-materialization/report.json`), so QAT is required. This layout has 512 FP32 scales and is a staged representation, not a packed INT3 release.
- Initial isolated QAT, seed `20260957`, learning rate `0.0001`, 200 epochs, produced matching `1 / 294,778` misses. Its single controlled continuation changed only learning rate to `0.00005` for 100 epochs and regressed to matching `3 / 294,778`; both immutable artifacts are rejected and this recipe will not be extended.

## Accepted INT3 scope 10

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1`

- Strategy change: output-projection granularity alone changed to independent scalar INT3 groups; accepted Q remains rowwise INT3, K/V remain group-2 INT3, and all predecessor tensors are unchanged.
- The parity test was observed red for the absent group-1 layout and then passed after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py`.
- Exhaustive direct materialization from the immutable scope-9 checkpoint passed with matching **0 / 294,778** fake-QAT and materialized-policy misses. No QAT ran; `attention.out_proj.weight` had zero trainable tensors.
- Immutable preflight/report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-direct-materialization/report.json` from `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt`.
- The output projection has 1,024 FP32 scales (one per scalar), so this is a policy-preserving staged representation, not a storage-efficient packed INT3 artifact or a full-parameter INT3 release.

## INT3 stage status

- Accepted staged scope is now scope 10: experts/output/router, scalar-group token and position tables, attention Q rowwise, K/V group-2, and output projection scalar-group INT3. Attention biases and norm tensors remain F32.
- Next ordered unresolved component: preflight `attention.in_proj_bias` under this accepted attention-projection predecessor. No packed INT3 artifact may be created until every declared tensor is accepted and independent packing/runtime/integrity gates exist.

## Accepted INT3 scope 11

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1`

- Scope: accepted INT3 scope 10 plus `attention.in_proj_bias` in independent scalar INT3 groups. `attention.out_proj.bias` and both norm tensors remain F32.
- The fake-QAT/materialized layout parity test passes: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py::test_int3_qk_group2_v_group2_out_group1_input_bias_group1_fake_qat_matches_materialized_runtime`.
- No QAT ran. Exhaustive direct materialization from the immutable scope-9 checkpoint passed with matching **0 / 294,778** fake-QAT and materialized-policy misses; zero tensors were trainable.
- Immutable preflight/report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-direct-materialization/report.json`.
- `attention.in_proj_bias` uses 96 FP32 scales (one per scalar), so this is a policy-preserving staged representation, not a storage-efficient packed INT3 artifact or a full-parameter INT3 release.

## INT3 stage status

- Accepted staged scope is now scope 11: experts/output/router, scalar-group token and position tables, attention Q rowwise, K/V group-2, scalar-group output projection, and scalar-group input-projection bias. Attention output bias and norm tensors remain F32.
- Decision: **advance**. The next ordered unresolved component is `attention.out_proj.bias`, requiring its own parity-tested direct-materialization preflight. No packed INT3 artifact may be created until every declared tensor is accepted and independent packing/runtime/integrity gates exist.

## Accepted INT3 scope 12

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1`

- Scope: accepted INT3 scope 11 plus `attention.out_proj.bias` in independent scalar INT3 groups. Both norm tensors remain F32.
- The new output-bias fake-QAT/materialized parity test was red for the absent layout method, then green after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py::test_int3_qk_group2_v_group2_out_group1_input_bias_group1_output_bias_group1_fake_qat_matches_materialized_runtime`.
- No QAT ran. Exhaustive direct materialization from the immutable scope-9 checkpoint passed with matching **0 / 294,778** fake-QAT and materialized-policy misses; zero tensors were trainable.
- Immutable preflight/report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-direct-materialization/report.json`.
- `attention.out_proj.bias` uses 32 FP32 scales (one per scalar), so this is a policy-preserving staged representation, not a storage-efficient packed INT3 artifact or a full-parameter INT3 release.

## INT3 stage status

- Accepted staged scope is now scope 12: experts/output/router, scalar-group token and position tables, attention Q rowwise, K/V group-2, scalar-group attention output projection and both attention biases. Norm tensors remain F32.
- Decision: **advance**. The next ordered unresolved component is `norm.weight`, requiring a new parity-tested direct-materialization preflight. No packed INT3 artifact may be created until every declared tensor is accepted and independent packing/runtime/integrity gates exist.

## Accepted INT3 scope 13

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-norm-weight-group1`

- Scope: accepted scope 12 plus `norm.weight` in independent scalar INT3 groups; `norm.bias` remains F32.
- The parity test passed: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py::test_int3_qk_group2_v_group2_out_group1_input_bias_group1_output_bias_group1_norm_weight_group1_fake_qat_matches_materialized_runtime`.
- No QAT ran. Exhaustive direct materialization from the immutable accepted Q/K/V checkpoint passed with matching **0 / 294,778** fake-QAT and materialized-policy misses; zero tensors were trainable.
- Immutable report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-norm-weight-group1-direct-materialization/report.json`.
- `norm.weight` uses 32 FP32 scales (one per scalar), so this is a policy-preserving staged representation, not a storage-efficient packed INT3 artifact.

## Accepted INT3 scope 14

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-norm-weight-group1-norm-bias-group1`

- Scope: accepted scope 13 plus `norm.bias` in independent scalar INT3 groups. All model parameters are now quantized under the explicitly recorded mixed layouts.
- The norm-bias parity test was observed red for absent methods, then green after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py::test_int3_qk_group2_v_group2_out_group1_input_bias_group1_output_bias_group1_norm_weight_group1_norm_bias_group1_fake_qat_matches_materialized_runtime`.
- No QAT ran. Exhaustive direct materialization from the immutable accepted Q/K/V checkpoint passed with matching **0 / 294,778** fake-QAT and materialized-policy misses; zero tensors were trainable.
- Immutable report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-norm-weight-group1-norm-bias-group1-direct-materialization/report.json`.
- The two norm tensors use 64 FP32 scales in total (one per scalar). This is a complete policy-quantized staged representation, **not** a packed INT3 deployment artifact: independent packing, runtime, and integrity gates remain unimplemented.

## INT3 stage status

- Accepted scope 14 covers every model parameter: expert matrices/biases, output tensors, router weight/bias, scalar-group embedding/position tables, attention Q/K/V/output/bias tensors, and scalar-group norm weight/bias.
- Decision: **change strategy** from staged QAT to representation work. No training process is active because all ordered parameter groups passed their exact direct/QAT gates. The next authorized work is parity-tested packed INT3 runtime/integrity design; no packed release exists or is claimed.

## Accepted packed INT3 runtime v1

`crystal-9-packed-int3-v1`

- Scope: the complete accepted mixed INT3 scope 14, sourced from `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt`. It preserves the recorded layouts: rowwise experts/output/Q, four-value router-weight groups, two-value K/V groups, and scalar groups elsewhere.
- Artifact: `artifacts/int3-packed-v1-preflight-20260924-retry1/crystal-9-int3-packed-v1.pt` (`55,489` bytes; SHA-256 `2bc68216b05d898f2728314bd467dc49cfd8390380e47747b2122174dc8574fc`; manifest integrity SHA-256 `5a27545c39fa2b327e25f8f62c4a16f4a820643cb3354ac36b4e97f2c115c58a`). It stores genuine signed three-bit codes packed low-bit-first with explicit FP32 scales.
- The independent `PackedInt3Policy` passed exhaustive packed-runtime evaluation with **0 / 294,778** legal-policy misses. Immutable report: `artifacts/int3-packed-v1-preflight-20260924-retry1/report.json`.
- Runtime/integrity gates: a payload-bit flip is rejected by manifest SHA-256 validation; malformed, repeated-square, oversized, and post-terminal histories (`!`, `aa`, `abcdefghi`, `adbecf`) each return `!`.
- TDD evidence: the runtime test was red for the absent module before implementation; the evaluator boundary test was red on an overlong encoded history and green after padding against encoded-token length. Targeted packed INT3 tests and the full suite pass (`107 passed`).
- This is a complete packed INT3 runtime candidate, not a claim that every scale is storage-optimal: scalar-group tensors retain one FP32 scale per scalar. Its public/release manifest work remains separate.

## INT3 stage status

- Accepted scope 14 covers every model parameter, and packed runtime/integrity candidate `crystal-9-packed-int3-v1` now passes its independent exhaustive gate.
- Decision: **advance** to release-manifest/provenance review. No training process is active; parameter QAT is complete and the next work is non-training release packaging verification.

## Staged packed INT3 release manifest v1

`crystal-9-packed-int3-v1`

- Release staging: `releases/huggingface-int3-v1/`; status is `staged-not-published`, with no external upload attempted.
- The staged artifact is an exact immutable copy of the independently accepted packed candidate: `55,489` bytes, SHA-256 `2bc68216b05d898f2728314bd467dc49cfd8390380e47747b2122174dc8574fc`, internal manifest SHA-256 `5a27545c39fa2b327e25f8f62c4a16f4a820643cb3354ac36b4e97f2c115c58a`.
- `release-manifest.json` records the source checkpoint and immutable acceptance report; `SHA256SUMS` covers the staged artifact, runtime, design, README, and manifest. The staged runtime loads successfully and independently exhaustively evaluates at **0 / 294,778** legal-policy misses.
- Decision: **blocked** on the declared publication blockers—approved hosted target and external upload approval. No training process is active and no additional ordered parameter or representation stage is authorized.

## INT2 direct-materialization status

`mixed-int2-rowwise-suffix-direct-materialization`

- INT2 is a new derivation from immutable `artifacts-fp32.pt`; accepted F32, INT4, and INT3 artifacts remain immutable predecessors and are not modified by this work.
- The independently parity-tested rowwise suffix scope (`experts.*.0.weight`, `experts.*.2.weight`, `output.weight`) was directly materialized from F32 with zero trainable tensors. It uses 2-bit row groups of 32 values and 589 FP32 scales; it is a staged representation only, with no packed runtime or release integrity gate.
- Exhaustive preflight: fake-QAT **98,739 / 294,778** misses; materialized **98,739 / 294,778** misses. Matching totals establish path parity, not correctness, so the candidate is rejected and must not enter QAT.
- Decision: **change strategy**. Preserve this rejected candidate and establish a different, independently parity-tested INT2 representation/granularity from immutable F32 before considering any isolated QAT.

## Rejected INT2 four-value-group suffix preflight

`mixed-int2-group4-suffix-direct-materialization`

- Strategy change: the immutable F32 source was quantized only in the suffix (`experts.*.0.weight`, `experts.*.2.weight`, `output.weight`) using independent contiguous four-value INT2 groups, rather than the rejected 32-value rowwise groups.
- TDD evidence: `tests/test_mixed_int2_suffix_group4_parity.py` was red for the absent fake-QAT/materialized layout, then green. The direct-preflight report test was likewise red for the absent runner, then green. Full suite: **114 passed**.
- Exhaustive direct-materialization preflight: matching fake-QAT/materialized totals of **10,264 / 294,778** policy misses. Immutable report: `artifacts/rejected/int2-group4-suffix-direct-materialization-20260924-rejected/report.json`.
- The layout uses 4,712 FP32 scales (four-value groups). It is a staged representation only, with no packed runtime or release integrity gate; its nonzero result is rejected and no QAT is authorized for this materially nonzero candidate.
- Decision: **change strategy**. The next candidate must use a distinct independently parity-tested INT2 scope or granularity from immutable F32; F32, accepted INT4, accepted INT3, and both rejected INT2 artifacts remain immutable.

