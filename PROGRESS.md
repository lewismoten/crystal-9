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
- This is material improvement over group4 but is not near the exact gate. The immutable checkpoint and report remain rejected evidence; this combined-table recipe will not be extended.

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
- Exact policy gate: **0 /294,778** misses in fake-QAT and separately materialized evaluation. Immutable report/checkpoint: `artifacts/int3-scalar-input-attention-q-k-group2-qat-200-seed20260952-lr1e-4/report.json` and `artifacts/int3-scalar-input-attention-q-k-group2-qat-200-seed20260952-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2.pt`.

## Active INT3 attention V four-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group4`

- Scope: accepted INT3 K group-2 predecessor plus only V rows (`attention.in_proj_weight[64:96]`) in contiguous four-value INT3 groups. Q remains rowwise INT3 and K remains group-2 INT3; all other tensors are frozen.
- New fake-QAT/materialized parity test: `tests/test_mixed_int3_attention_q_k_group2_v_group4_parity.py` was red for the absent layout and passes after implementation.
- Exhaustive direct materialization from the accepted K checkpoint produced matching `63 /294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-v-group4-direct-materialization/report.json`), so QAT is required. V has 1,024 FP32 scales and is a staged representation, not a packed INT3 release.
- Active recipe: isolated 200 epochs, seed `20260953`, learning rate `0.0001`, batch size `1024`; only V is trainable, while Q/K slices and all predecessor tensors are SHA-256 asserted frozen.

## Active INT3 attention V four-value-group controlled continuation

`mixed-int3-scalar-input-attention-q-k-group2-v-group4`

- The initial isolated V-only candidate completed at matching `7 /294,778` fake-QAT/materialized-policy misses. This is near the exact gate and immutable evidence, not accepted.
- Decision: **continue** once from its best checkpoint, changing only learning rate from `0.0001` to `0.00005`; same seed `20260953`, 100 epochs, batch size `1024`, and only V rows trainable. Q/K slices and all non-projection predecessor tensors remain SHA-256 asserted frozen.
- Running artifact directory: `artifacts/int3-scalar-input-attention-q-k-group2-v-group4-qat-continue-100-seed20260953-lr5e-5/`. Acceptance remains exactly `0 /294,778` in both paths. A regression or nonzero total rejects this recipe; the next action will be a parity-tested V granularity change.

## Rejected INT3 attention V four-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group4`

- Initial isolated QAT reached matching `7 /294,778` misses; its one controlled continuation changed only learning rate to `0.00005` for 100 epochs and regressed to matching `14 /294,778` misses.
- Both group-4 V QAT artifacts are immutable rejected evidence. Do not extend this recipe again.

## Active INT3 attention V two-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group2`

- Strategy change: only V quantization granularity changes from rejected group-4 V to independent contiguous two-value INT3 groups per row. Accepted Q remains rowwise INT3, accepted K remains group-2 INT3, and all other predecessor tensors remain frozen.
- The fake-QAT/materialized parity test was observed red for absent group-2 V methods and passes after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group4_parity.py`. The isolated runner test asserts exact Q/K and all non-projection predecessor tensor identity; AdamW uses zero weight decay while Q/K gradients are masked.
- Exhaustive direct materialization from the accepted K checkpoint produced matching `10 /294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-v-group2-direct-materialization/report.json`), so QAT is required. V has 512 FP32 scales and is a staged representation, not a packed INT3 release.
- Active recipe: isolated 200 epochs, seed `20260954`, learning rate `0.0001`, batch size `1024`; only V is trainable, while Q/K slices and all non-projection predecessor tensors are SHA-256 asserted frozen. Acceptance remains exactly `0 /294,778` in both paths.

## Accepted INT3 scope 9

`mixed-int3-scalar-input-attention-q-k-group2-v-group2`

- Scope: accepted INT3 scope 8 plus V rows (`attention.in_proj_weight[64:96]`) in independent contiguous two-value INT3 groups. Q remains rowwise INT3 and K remains group-2 INT3; attention output projection/bias and norm tensors remain F32.
- Direct materialization missed `10 /294,778`; isolated V-only QAT trained 200 epochs at seed `20260954`, learning rate `0.0001`, batch size `1024`, with Q/K slices and every non-projection predecessor tensor SHA-256 asserted frozen.
- Exact policy gate: **0 /294,778** misses in fake-QAT and separately materialized evaluation. Immutable report/checkpoint: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/report.json` and `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt`.
- V uses 512 FP32 scales (group size 2), so this remains a staged representation, not a packed INT3 release.

## Active INT3 attention output-projection rowwise candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out`

- Scope: accepted INT3 scope 9 plus only `attention.out_proj.weight` in rowwise INT3. Q/K/V accepted layouts remain frozen, as do both attention biases and all other predecessor tensors.
- New fake-QAT/materialized parity test was observed red before implementation and now passes: `tests/test_mixed_int3_attention_q_k_group2_v_group4_parity.py`. The isolated runner test asserts only the attention output-projection weight can change.
- Exhaustive direct materialization from accepted scope 9 missed matching `2,086 /294,778` fake-QAT/materialized legal policies (`artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-direct-materialization/report.json`); QAT is required.
- Initial isolated QAT: 200 epochs, seed `20260955`, learning rate `0.0001`, batch size `1024`, trained only `attention.out_proj.weight` with zero optimizer weight decay and SHA-256 inventory for every frozen tensor. It completed at matching `275 /294,778` fake-QAT/materialized misses and is immutable rejected evidence, not a candidate to extend.

## Active INT3 attention output-projection four-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group4`

- Strategy change: only the output-projection quantization granularity changes from the rejected rowwise recipe to independent contiguous four-value INT3 groups; Q/K/V and every other predecessor tensor remain frozen.
- The new fake-QAT/materialized parity test was observed red before implementation and is green: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py`. The isolated-runner test asserts every tensor other than `attention.out_proj.weight` retains exact identity.
- Exhaustive direct materialization from accepted scope 9 produced matching `145 /294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group4-direct-materialization/report.json`), so QAT is required. This layout has 256 FP32 scales and is a staged representation, not a packed INT3 release.
- Initial isolated QAT: 200 epochs, seed `20260956`, learning rate `0.0001`, batch size `1024`, trained only `attention.out_proj.weight` with zero optimizer weight decay and SHA-256 assertions for every frozen predecessor tensor. It reduced matching fake-QAT/materialized misses from `145` to `7 /294,778`; this is immutable evidence, not acceptance.
- Decision: **change strategy**. The one allowed group-4 continuation regressed from matching `7` to `12 /294,778` misses, so both group-4 QAT artifacts are immutable rejected evidence; do not extend them again.

## Active INT3 attention output-projection two-value-group candidate

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group2`

- Strategy change: only `attention.out_proj.weight` granularity changes from rejected group-4 to independent contiguous two-value INT3 groups. Accepted Q/K/V and every other predecessor tensor remain frozen.
- The parity test was observed red for the absent group-2 output layout, then green after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py`. The isolated runner test asserts every tensor other than `attention.out_proj.weight` retains exact identity.
- Exhaustive direct materialization from accepted scope 9 produced matching `32 /294,778` fake-QAT/materialized misses (`artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group2-direct-materialization/report.json`), so QAT is required. This layout has 512 FP32 scales and is a staged representation, not a packed INT3 release.
- Initial isolated QAT, seed `20260957`, learning rate `0.0001`, 200 epochs, produced matching `1 /294,778` misses. Its single controlled continuation changed only learning rate to `0.00005` for 100 epochs and regressed to matching `3 /294,778`; both immutable artifacts are rejected and this recipe will not be extended.

## Accepted INT3 scope 10

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1`

- Strategy change: output-projection granularity alone changed to independent scalar INT3 groups; accepted Q remains rowwise INT3, K/V remain group-2 INT3, and all predecessor tensors are unchanged.
- The parity test was observed red for the absent group-1 layout and then passed after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py`.
- Exhaustive direct materialization from the immutable scope-9 checkpoint passed with matching **0 /294,778** fake-QAT and materialized-policy misses. No QAT ran; `attention.out_proj.weight` had zero trainable tensors.
- Immutable preflight/report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-direct-materialization/report.json` from `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt`.
- The output projection has 1,024 FP32 scales (one per scalar), so this is a policy-preserving staged representation, not a storage-efficient packed INT3 artifact or a full-parameter INT3 release.

## INT3 stage status

- Accepted staged scope is now scope 10: experts/output/router, scalar-group token and position tables, attention Q rowwise, K/V group-2, and output projection scalar-group INT3. Attention biases and norm tensors remain F32.
- Next ordered unresolved component: preflight `attention.in_proj_bias` under this accepted attention-projection predecessor. No packed INT3 artifact may be created until every declared tensor is accepted and independent packing/runtime/integrity gates exist.

## Accepted INT3 scope 11

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1`

- Scope: accepted INT3 scope 10 plus `attention.in_proj_bias` in independent scalar INT3 groups. `attention.out_proj.bias` and both norm tensors remain F32.
- The fake-QAT/materialized layout parity test passes: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py::test_int3_qk_group2_v_group2_out_group1_input_bias_group1_fake_qat_matches_materialized_runtime`.
- No QAT ran. Exhaustive direct materialization from the immutable scope-9 checkpoint passed with matching **0 /294,778** fake-QAT and materialized-policy misses; zero tensors were trainable.
- Immutable preflight/report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-direct-materialization/report.json`.
- `attention.in_proj_bias` uses 96 FP32 scales (one per scalar), so this is a policy-preserving staged representation, not a storage-efficient packed INT3 artifact or a full-parameter INT3 release.

## INT3 stage status

- Accepted staged scope is now scope 11: experts/output/router, scalar-group token and position tables, attention Q rowwise, K/V group-2, scalar-group output projection, and scalar-group input-projection bias. Attention output bias and norm tensors remain F32.
- Decision: **advance**. The next ordered unresolved component is `attention.out_proj.bias`, requiring its own parity-tested direct-materialization preflight. No packed INT3 artifact may be created until every declared tensor is accepted and independent packing/runtime/integrity gates exist.

## Accepted INT3 scope 12

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1`

- Scope: accepted INT3 scope 11 plus `attention.out_proj.bias` in independent scalar INT3 groups. Both norm tensors remain F32.
- The new output-bias fake-QAT/materialized parity test was red for the absent layout method, then green after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py::test_int3_qk_group2_v_group2_out_group1_input_bias_group1_output_bias_group1_fake_qat_matches_materialized_runtime`.
- No QAT ran. Exhaustive direct materialization from the immutable scope-9 checkpoint passed with matching **0 /294,778** fake-QAT and materialized-policy misses; zero tensors were trainable.
- Immutable preflight/report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-direct-materialization/report.json`.
- `attention.out_proj.bias` uses 32 FP32 scales (one per scalar), so this is a policy-preserving staged representation, not a storage-efficient packed INT3 artifact or a full-parameter INT3 release.

## INT3 stage status

- Accepted staged scope is now scope 12: experts/output/router, scalar-group token and position tables, attention Q rowwise, K/V group-2, scalar-group attention output projection and both attention biases. Norm tensors remain F32.
- Decision: **advance**. The next ordered unresolved component is `norm.weight`, requiring a new parity-tested direct-materialization preflight. No packed INT3 artifact may be created until every declared tensor is accepted and independent packing/runtime/integrity gates exist.

## Accepted INT3 scope 13

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-norm-weight-group1`

- Scope: accepted scope 12 plus `norm.weight` in independent scalar INT3 groups; `norm.bias` remains F32.
- The parity test passed: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py::test_int3_qk_group2_v_group2_out_group1_input_bias_group1_output_bias_group1_norm_weight_group1_fake_qat_matches_materialized_runtime`.
- No QAT ran. Exhaustive direct materialization from the immutable accepted Q/K/V checkpoint passed with matching **0 /294,778** fake-QAT and materialized-policy misses; zero tensors were trainable.
- Immutable report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-norm-weight-group1-direct-materialization/report.json`.
- `norm.weight` uses 32 FP32 scales (one per scalar), so this is a policy-preserving staged representation, not a storage-efficient packed INT3 artifact.

## Accepted INT3 scope 14

`mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-norm-weight-group1-norm-bias-group1`

- Scope: accepted scope 13 plus `norm.bias` in independent scalar INT3 groups. All model parameters are now quantized under the explicitly recorded mixed layouts.
- The norm-bias parity test was observed red for absent methods, then green after minimal implementation: `tests/test_mixed_int3_attention_q_k_group2_v_group2_out_group4_parity.py::test_int3_qk_group2_v_group2_out_group1_input_bias_group1_output_bias_group1_norm_weight_group1_norm_bias_group1_fake_qat_matches_materialized_runtime`.
- No QAT ran. Exhaustive direct materialization from the immutable accepted Q/K/V checkpoint passed with matching **0 /294,778** fake-QAT and materialized-policy misses; zero tensors were trainable.
- Immutable report: `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-out-group1-input-bias-group1-output-bias-group1-norm-weight-group1-norm-bias-group1-direct-materialization/report.json`.
- The two norm tensors use 64 FP32 scales in total (one per scalar). This is a complete policy-quantized staged representation, **not** a packed INT3 deployment artifact: independent packing, runtime, and integrity gates remain unimplemented.

## INT3 stage status

- Accepted scope 14 covers every model parameter: expert matrices/biases, output tensors, router weight/bias, scalar-group embedding/position tables, attention Q/K/V/output/bias tensors, and scalar-group norm weight/bias.
- Decision: **change strategy** from staged QAT to representation work. No training process is active because all ordered parameter groups passed their exact direct/QAT gates. The next authorized work is parity-tested packed INT3 runtime/integrity design; no packed release exists or is claimed.

## Accepted packed INT3 runtime v1

`crystal-9-packed-int3-v1`

- Scope: the complete accepted mixed INT3 scope 14, sourced from `artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt`. It preserves the recorded layouts: rowwise experts/output/Q, four-value router-weight groups, two-value K/V groups, and scalar groups elsewhere.
- Artifact: `artifacts/int3-packed-v1-preflight-20260924-retry1/crystal-9-int3-packed-v1.pt` (`55,489` bytes; SHA-256 `2bc68216b05d898f2728314bd467dc49cfd8390380e47747b2122174dc8574fc`; manifest integrity SHA-256 `5a27545c39fa2b327e25f8f62c4a16f4a820643cb3354ac36b4e97f2c115c58a`). It stores genuine signed three-bit codes packed low-bit-first with explicit FP32 scales.
- The independent `PackedInt3Policy` passed exhaustive packed-runtime evaluation with **0 /294,778** legal-policy misses. Immutable report: `artifacts/int3-packed-v1-preflight-20260924-retry1/report.json`.
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
- `release-manifest.json` records the source checkpoint and immutable acceptance report; `SHA256SUMS` covers the staged artifact, runtime, design, README, and manifest. The staged runtime loads successfully and independently exhaustively evaluates at **0 /294,778** legal-policy misses.
- Decision: **blocked** on the declared publication blockers—approved hosted target and external upload approval. No training process is active and no additional ordered parameter or representation stage is authorized.

## INT2 direct-materialization status

`mixed-int2-rowwise-suffix-direct-materialization`

- INT2 is a new derivation from immutable `artifacts-fp32.pt`; accepted F32, INT4, and INT3 artifacts remain immutable predecessors and are not modified by this work.
- The independently parity-tested rowwise suffix scope (`experts.*.0.weight`, `experts.*.2.weight`, `output.weight`) was directly materialized from F32 with zero trainable tensors. It uses 2-bit row groups of 32 values and 589 FP32 scales; it is a staged representation only, with no packed runtime or release integrity gate.
- Exhaustive preflight: fake-QAT **98,739 /294,778** misses; materialized **98,739 /294,778** misses. Matching totals establish path parity, not correctness, so the candidate is rejected and must not enter QAT.
- Decision: **change strategy**. Preserve this rejected candidate and establish a different, independently parity-tested INT2 representation/granularity from immutable F32 before considering any isolated QAT.

## Rejected INT2 four-value-group suffix preflight

`mixed-int2-group4-suffix-direct-materialization`

- Strategy change: the immutable F32 source was quantized only in the suffix (`experts.*.0.weight`, `experts.*.2.weight`, `output.weight`) using independent contiguous four-value INT2 groups, rather than the rejected 32-value rowwise groups.
- TDD evidence: `tests/test_mixed_int2_suffix_group4_parity.py` was red for the absent fake-QAT/materialized layout, then green. The direct-preflight report test was likewise red for the absent runner, then green. Full suite: **114 passed**.
- Exhaustive direct-materialization preflight: matching fake-QAT/materialized totals of **10,264 /294,778** policy misses. Immutable report: `artifacts/rejected/int2-group4-suffix-direct-materialization-20260924-rejected/report.json`.
- The layout uses 4,712 FP32 scales (four-value groups). It is a staged representation only, with no packed runtime or release integrity gate; its nonzero result is rejected and no QAT is authorized for this materially nonzero candidate.
- Decision: **change strategy**. The next candidate must use a distinct independently parity-tested INT2 scope or granularity from immutable F32; F32, accepted INT4, accepted INT3, and both rejected INT2 artifacts remain immutable.

## Accepted INT2 scalar suffix plus output-bias preflight

`mixed-int2-scalar-suffix-output-bias-direct-materialization`

- The immutable F32 source (`artifacts-fp32.pt`, SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with scalar-group INT2 applied only to `experts.*.0.weight`, `experts.*.2.weight`, `output.weight`, and `output.bias`; zero tensors were trainable and no QAT ran.
- TDD evidence: both fake-QAT/materialized parity and the direct-preflight report runner were red for their absent API, then green; full suite: **117 passed**.
- Exhaustive direct-materialization gate passed: **0 /294,778** legal-policy misses in fake-QAT and independently materialized runtime. Immutable report: `artifacts/int2-scalar-suffix-output-bias-direct-materialization-20260924/report.json`.
- The representation has 18,861 FP32 scalar scales and is **not storage-efficient**; it is staged research only, not a packed INT2 runtime or release.
- Decision: **advance**. The next ordered direct-materialization candidate is scalar-group INT2 `router.weight` from this accepted scope, with a distinct parity test and exhaustive gate before any QAT.

## Accepted INT2 scalar suffix, output-bias, and router-weight preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-direct-materialization`

- Immutable F32 source (`artifacts-fp32.pt`, SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with scalar-group INT2 on `experts.*.0.weight`, `experts.*.2.weight`, `output.weight`, `output.bias`, and `router.weight`. Zero tensors were trainable; no QAT ran.
- TDD evidence: the new fake-QAT/materialized parity test failed because the layout was absent, then passed after minimal implementation. The direct-preflight test likewise failed for its absent runner, then passed. Full suite: **119 passed**.
- Exhaustive direct-materialization gate passed: **0 /294,778** legal-policy misses in fake-QAT and independently materialized runtime. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-direct-materialization-20260924/report.json`.
- The representation has 19,149 FP32 scalar scales and is **not storage-efficient**; it remains staged research only, not a packed INT2 runtime or release.
- Decision: **advance**. The next ordered direct-materialization candidate is scalar-group INT2 `router.bias` from this accepted scope, with a distinct parity test and exhaustive gate before any QAT.

## Accepted INT2 scalar suffix, output bias, router weight, and router bias preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-direct-materialization`

- Immutable F32 source (`artifacts-fp32.pt`, SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with scalar-group INT2 on `experts.*.0.weight`, `experts.*.2.weight`, `output.weight`, `output.bias`, `router.weight`, and `router.bias`. Zero tensors were trainable; no QAT ran.
- TDD evidence: the direct-preflight runner test failed for the absent API, then passed after minimal implementation; the independently created fake-QAT/materialized parity test passes. Full suite: **121 passed**.
- Exhaustive direct-materialization gate passed: **0 /294,778** legal-policy misses in fake-QAT and independently materialized runtime. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-direct-materialization-20260924/report.json`.
- The representation has 19,158 FP32 scalar scales and is **not storage-efficient**; it remains staged research only, not a packed INT2 runtime or release.
- Decision: **advance**. The next ordered direct-materialization candidate is scalar-group INT2 expert biases (`experts.*.0.bias`, `experts.*.2.bias`) from this accepted scope, with a distinct parity test and exhaustive gate before any QAT; `embedding.weight`, attention, and norm tensors remain F32.

## Accepted INT2 scalar suffix through expert-bias preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-direct-materialization`

- Immutable F32 source (`artifacts-fp32.pt`, SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with scalar-group INT2 on `experts.*.0.weight`, `experts.*.2.weight`, `output.weight`, `output.bias`, `router.weight`, `router.bias`, `experts.*.0.bias`, and `experts.*.2.bias`; zero tensors were trainable and no QAT ran.
- TDD evidence: the direct-preflight runner test failed for its absent API, then passed after minimal implementation; the preceding independent fake-QAT/materialized parity test passes. Full suite: **123 passed**.
- Exhaustive direct-materialization gate passed: **0 /294,778** legal-policy misses in fake-QAT and independently materialized runtime. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-direct-materialization-20260924/report.json`.
- The representation has 19,734 FP32 scalar scales and is **not storage-efficient**; it remains staged research only, not a packed INT2 runtime or release.
- Decision: **advance**. The next ordered direct-materialization candidate is scalar-group INT2 `position.weight` from this accepted scope, with a distinct parity test and exhaustive gate before any QAT; `embedding.weight`, attention, and norm tensors remain F32.

## Accepted INT2 scalar suffix through position preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-direct-materialization`

- Immutable F32 source (`artifacts-fp32.pt`, SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with scalar-group INT2 on `experts.*.0.weight`, `experts.*.2.weight`, `output.weight`, `output.bias`, `router.weight`, `router.bias`, `experts.*.0.bias`, `experts.*.2.bias`, and `position.weight`; zero tensors were trainable and no QAT ran.
- TDD evidence: the fake-QAT/materialized parity test was red for the absent layout, then green; the direct-preflight test was red for the absent runner, then green.
- Exhaustive direct-materialization gate passed: **0 /294,778** legal-policy misses in fake-QAT and independently materialized runtime. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-direct-materialization/report.json`.
- The representation has 20,022 FP32 scalar scales and is **not storage-efficient**; it remains staged research only, not a packed INT2 runtime or release.
- Decision: **advance**. The next ordered direct-materialization candidate is scalar-group INT2 `embedding.weight` from this accepted scope, with a distinct parity test and exhaustive gate before any QAT; attention and norm tensors remain F32.

## Accepted INT2 scalar suffix through both input tables preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-direct-materialization`

- Immutable F32 source (`artifacts-fp32.pt`, SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with scalar-group INT2 on the complete previous scope plus `embedding.weight`; zero tensors were trainable and no QAT ran.
- TDD evidence: the fake-QAT/materialized parity test failed for the absent layout and then passed after minimal implementation; the direct-preflight runner test likewise failed for its absent API and then passed. Full suite: **127 passed**.
- Exhaustive direct-materialization gate passed: **0 /294,778** legal-policy misses in fake-QAT and independently materialized runtime. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-direct-materialization-20260924/report.json`.
- The representation has 20,438 FP32 scalar scales and is **not storage-efficient**; it remains staged research only, not a packed INT2 runtime or release.
- Decision: **advance**. The next ordered direct-materialization candidate is scalar-group INT2 attention Q rows (`attention.in_proj_weight[:32]`) from this accepted scope, with a distinct parity test and exhaustive gate before any QAT; K/V/output projection, attention biases, and norm tensors remain F32.

## Rejected INT2 scalar input-table plus attention-Q preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-direct-materialization`

- The scalar INT2 input-table scope was independently parity-tested with Q rows (`attention.in_proj_weight[:32]`) at rowwise INT2. The parity and direct-preflight tests were red for absent APIs and green after minimal implementations.
- Exhaustive direct materialization from immutable F32 produced matching **3,954 /294,778** fake-QAT/materialized-policy misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-direct-materialization-20260924/report.json`.
- This materially nonzero candidate is rejected. No QAT was started; matching outcomes establish parity, not policy correctness. Preserve it as rejected evidence.
- Decision: **change strategy**. There is no active INT2 training job: the next attention candidate must change scope or quantization granularity and have its own red-to-green parity test; it must not extend this rowwise-Q direct candidate.

## Accepted INT2 scalar-Q direct-materialization preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-direct-materialization`

- Strategy change: Q rows (`attention.in_proj_weight[:32]`) use independent scalar INT2 groups; the prior rowwise-Q candidate remains rejected and immutable.
- TDD evidence: independent fake-QAT/materialized parity and direct-preflight tests each failed for absent APIs, then passed after minimal implementation.
- Immutable F32 source `artifacts-fp32.pt` (SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`), zero trainable tensors, no QAT.
- Exhaustive direct-materialization gate: **0 / 294,778** fake-QAT misses and **0 / 294,778** independently materialized-runtime misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-direct-materialization/report.json`.
- The staged representation has 21,462 FP32 scalar scales and is not storage-efficient; it is neither packed nor a full INT2 release.
- Decision: **advance**. The next ordered candidate is scalar-group INT2 K rows (`attention.in_proj_weight[32:64]`), with a fresh red-to-green parity test and exhaustive direct-materialization gate before any QAT.

## Accepted INT2 scalar-Q/K direct-materialization preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-direct-materialization`

- Strategy change from the rejected rowwise-Q scope: the immutable F32 source (`artifacts-fp32.pt`, SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) now has both Q (`attention.in_proj_weight[:32]`) and K (`attention.in_proj_weight[32:64]`) rows represented as independent scalar INT2 groups. No tensors were trainable and no QAT ran.
- TDD evidence: the K parity test passes, and the direct-preflight runner test was observed red for its absent runner then green after its minimal implementation.
- Exhaustive direct-materialization gate: **0 / 294,778** fake-QAT misses and **0 / 294,778** independently materialized-runtime misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-direct-materialization/report.json`.
- The staged representation has 22,486 FP32 scalar scales and is not storage-efficient; it is neither packed nor a full INT2 release.
- Decision: **advance**. The next ordered candidate is scalar-group INT2 V rows (`attention.in_proj_weight[64:96]`), requiring its own fresh red-to-green parity test and exhaustive direct-materialization gate before any QAT.

## Accepted INT2 scalar-Q/K/V direct-materialization preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-direct-materialization`

- Immutable F32 source `artifacts-fp32.pt` (SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with the accepted scalar suffix/input/Q/K scope plus scalar INT2 V rows (`attention.in_proj_weight[64:96]`). Zero tensors were trainable; no QAT ran.
- TDD evidence: the direct-preflight runner test failed because the V runner API was absent, then passed after its minimal implementation; the independently created fake-QAT/materialized parity test passes. Full suite: **135 passed**.
- Exhaustive direct-materialization gate passed: **0 /294,778** fake-QAT misses and **0 /294,778** independently materialized-runtime misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-direct-materialization/report.json`.
- The staged representation has 23,510 FP32 scalar scales and is not storage-efficient; it is neither packed nor a full INT2 release.
- Decision: **advance**. The next ordered candidate is scalar-group INT2 attention output-projection weights (`attention.out_proj.weight`) from this accepted scope, with fresh red-to-green parity and exhaustive direct-materialization tests before any QAT.

## Accepted INT2 scalar-Q/K/V/output-projection direct-materialization preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-direct-materialization`

- Immutable F32 source `artifacts-fp32.pt` (SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with the accepted scalar suffix/input/Q/K/V scope plus scalar INT2 `attention.out_proj.weight`. Zero tensors were trainable; no QAT ran.
- TDD evidence: Q/K/V/output-projection parity and preflight tests both failed for absent APIs, then passed after minimal implementations. Full suite: **137 passed**.
- Exhaustive direct-materialization gate passed: **0 /294,778** fake-QAT misses and **0 /294,778** independently materialized-runtime misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-direct-materialization/report.json`.
- The staged representation has 24,534 FP32 scalar scales and is not storage-efficient; it is neither packed nor a full INT2 release.
- Decision: **advance**. The next ordered candidate is scalar-group INT2 attention input-projection bias (`attention.in_proj_bias`), with fresh red-to-green parity and exhaustive direct-materialization tests before any QAT.

## Accepted INT2 scalar-Q/K/V/output-projection/input-bias direct-materialization preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-direct-materialization`

- Immutable F32 source `artifacts-fp32.pt` (SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with the accepted scalar suffix/input/Q/K/V/output-projection scope plus scalar INT2 `attention.in_proj_bias`. Zero tensors were trainable; no QAT ran.
- TDD evidence: parity and direct-preflight tests both failed for their absent APIs, then passed after minimal implementations.
- Exhaustive direct-materialization gate passed: **0 / 294,778** fake-QAT misses and **0 / 294,778** independently materialized-runtime misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-direct-materialization/report.json`.
- The staged representation has 24,630 FP32 scalar scales and is not storage-efficient; it is neither packed nor a full INT2 release.
- Decision: **advance**. The next ordered candidate is scalar-group INT2 attention output-projection bias (`attention.out_proj.bias`), with fresh red-to-green parity and exhaustive direct-materialization tests before any QAT.

## Accepted INT2 scalar attention output-bias preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-out-bias-group1-direct-materialization`

- Immutable F32 source `artifacts-fp32.pt` (SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with the accepted scalar scope plus scalar INT2 `attention.out_proj.bias`. Zero tensors were trainable; no QAT ran.
- TDD evidence: the output-bias fake-QAT/materialized parity test failed for the absent layout, then passed after minimal implementation; the direct-preflight runner test failed for its absent API, then passed after implementation.
- Exhaustive direct-materialization gate passed: **0 / 294,778** fake-QAT misses and **0 / 294,778** independently materialized-runtime misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-out-bias-group1-direct-materialization/report.json`.
- The staged representation has 24,662 FP32 scalar scales and is not storage-efficient; it is neither packed nor a full INT2 release.
- Decision: **advance**. The next ordered candidate is scalar-group INT2 `norm.weight`; its parity test has completed red-to-green and its exhaustive direct-materialization preflight is next. No QAT is authorized unless that preflight is nonzero.

## Accepted INT2 scalar norm-weight preflight

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-out-bias-group1-norm-weight-group1-direct-materialization`

- Immutable F32 source `artifacts-fp32.pt` (SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with the accepted scalar scope plus scalar INT2 `norm.weight`. Zero tensors were trainable and no QAT ran.
- TDD evidence: the norm-weight direct-preflight test failed for its absent runner API, then passed after minimal implementation; the existing fake-QAT/materialized parity test passes.
- Exhaustive direct-materialization gate passed: **0 / 294,778** fake-QAT misses and **0 / 294,778** independently materialized-runtime misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-out-bias-group1-norm-weight-group1-direct-materialization/report.json`.
- The staged representation has 24,694 FP32 scalar scales and is **not storage-efficient**; it is neither packed nor a full INT2 release.
- Decision: **advance**. The next ordered candidate is scalar-group INT2 `norm.bias`. Its fresh fake-QAT/materialized parity test was red for absent methods and is now green; exhaustive direct materialization is next. No QAT is authorized unless that preflight is nonzero.

## Accepted full scalar-group INT2 behavioral proof

`mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-out-bias-group1-norm-weight-group1-norm-bias-group1-direct-materialization`

- Immutable F32 source `artifacts-fp32.pt` (SHA-256 `e5e3aa5eee628c3d3911acabfc9b31eac093f5ec4c8435773537c34312b9399c`) was evaluated with scalar-group INT2 across every model parameter, including `norm.bias`. Zero tensors were trainable; no QAT ran.
- TDD evidence: the norm-bias direct-preflight test failed for its absent runner API, then passed after minimal implementation; its independently created fake-QAT/materialized parity test passes.
- Exhaustive direct-materialization gate passed: **0 / 294,778** fake-QAT misses and **0 / 294,778** independently materialized-runtime misses. Immutable report: `artifacts/int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-out-bias-group1-norm-weight-group1-norm-bias-group1-direct-materialization/report.json`.
- This first behaviorally exact full-parameter INT2 proof has 24,726 FP32 scalar scales. It is **not storage-efficient**, packed, or a deployable INT2 release.
- Decision: **change strategy**. Behavioral feasibility is proven. Any scale compression or packed-runtime work must be a distinct hierarchy-quantization candidate with its own red-to-green parity, packing, runtime, and integrity gates; this accepted proof remains immutable.

## Accepted packed scalar INT2 scale-compression candidate

`complete-scalar-group-int2-packed-fp16-scales`

- The immutable full scalar-group INT2 proof was independently exported as genuine low-bit-first packed INT2 codes, with all 24,726 scalar dequantization scales stored as FP16. The F32 source remains authoritative; accepted F32, INT4, INT3, and the FP32-scale INT2 research artifact are unchanged.
- TDD evidence: `tests/test_packed_int2_runtime.py` was red for the absent packed runtime, then green; `tests/test_packed_int2_fp16_scales.py` was red for the absent scale-dtype API, then green. Full suite: **149 passed**.
- Independent packed runtime exhaustive gate passed: **0 / 294,778** legal-policy misses. Immutable report: `artifacts/int2-packed-scalar-fp16-scales-preflight-20260924/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp16-scales-preflight-20260924/crystal-9-int2-packed-scalar-fp16-scales.pt` (`87,657` bytes; SHA-256 `916f9b8ff9ee2d6a02c920987df1e21f82a4ce01dc64c9d83c914bf9ea5f8f1a`; manifest integrity SHA-256 `f4619917528d92faf7393a564728be9e70dcc56cece9cea1a37b59e28c0524ec`).
- This is a parity-validated hierarchy-quantization research candidate, not a claimed compact or deployable INT2 release: it retains one scale per scalar and has no release-manifest or distribution integrity gate.
- Decision: **advance**. The next bounded representation candidate is a separately parity-tested lower-precision scale encoding; it must preserve this artifact and pass its own exhaustive packed-runtime and integrity gates.

## Accepted packed scalar INT2 FP8-scale candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales`

- The immutable full scalar-group INT2 proof was independently exported as genuine low-bit-first packed INT2 codes with all 24,726 scalar dequantization scales stored as `float8_e4m3fn`. The F32 source, accepted INT4/INT3 artifacts, FP32-scale INT2 proof, and FP16-scale candidate remain unchanged.
- TDD evidence: the FP8-scale runtime test was red because the original integrity serializer could not encode Float8 scales; it passed after byte-level digest serialization and Float32 decode promotion were added. A packed-runtime evaluator test was red for its absent API and then green. Full suite: **151 passed**.
- Independent packed runtime exhaustive gate passed: **0 / 294,778** legal-policy misses. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-preflight-20260924/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-preflight-20260924/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales.pt` (`64,525` bytes; SHA-256 `0908e9953a434f0ab5d78df8f6dbc8d4c5eb1c24db7e9e013bd7bb5105768235`; manifest integrity SHA-256 `d438aca4987c0b2efd6b71e6776c42d919e4a8fd1caaa5a545490860aedd9d26`).
- This is a parity-validated hierarchy-quantization research candidate, not a compact or deployable INT2 release: it still retains one scale per scalar and has no release-manifest or distribution integrity gate.
- Decision: **change strategy**. Lower scalar-scale precision is now exhausted for this FP8 type; any further compression must be a distinct parity-tested shared/hierarchical scale-layout candidate. No training process is active.

## Rejected packed INT2 two-value shared-scale candidate

`complete-int2-two-value-shared-fp8-scales-research`

- A distinct hierarchy candidate quantized the immutable F32 source into low-bit-first INT2 codes with one shared `float8_e4m3fn` scale for each contiguous two-value group. It has `12,364` scales rather than the scalar proof's `24,726`; accepted F32, INT4, INT3, scalar INT2 proof, FP16-scale, and scalar-FP8-scale artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_group2_scales.py` first failed because the group-2 runtime/export module was absent, then passed after the minimal runtime implementation. Targeted packed-INT2 tests pass and the full suite passes: **152 passed**.
- The packed artifact's payload integrity gate rejects a one-bit packed-payload flip. Exhaustive independent runtime evaluation failed the exact policy gate with **28,833 / 294,778** legal-policy misses. Immutable rejected report: `artifacts/rejected/int2-packed-group2-fp8-scales-preflight-20260924/report.json`.
- Rejected artifact: `artifacts/rejected/int2-packed-group2-fp8-scales-preflight-20260924/crystal-9-int2-packed-group2-fp8-scales.pt` (`50,307` bytes; SHA-256 `321b02cd62cb33ddb1fc8ac1b61671fe3e976346c93ea5d5a61bb060c49ac7fc`; manifest integrity SHA-256 `5a262ac501d1502136b880ee734730ba2607c93538738a7223748c3c38fc681e`). It is not a release or accepted representation.
- Decision: **change strategy**. Do not train or extend this direct-materialization representation. A future scale-compression candidate must alter the hierarchy design and independently establish parity before exhaustive evaluation; no model process is active.

## Rejected packed scalar INT2 FP8-E5M2-scale candidate

`complete-scalar-group-int2-packed-fp8-e5m2-scales`

- A distinct scalar-scale encoding candidate exported the immutable F32 source into low-bit-first packed INT2 codes with all 24,726 scalar dequantization scales stored as `float8_e5m2`. Accepted F32, INT4, INT3, the scalar INT2 proof, and the FP16/FP8-E4M3FN scale candidates remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_e5m2_preflight.py` was red because its dedicated preflight runner did not exist, then green after minimal implementation. Targeted packed-INT2 tests: **6 passed**.
- The exhaustive independent packed runtime gate produced **52 / 294,778** legal-policy misses. Immutable rejected report: `artifacts/rejected/int2-packed-scalar-fp8-e5m2-scales-preflight-20260924/report.json`.
- Rejected artifact: `artifacts/rejected/int2-packed-scalar-fp8-e5m2-scales-preflight-20260924/crystal-9-int2-packed-scalar-fp8-e5m2-scales.pt` (`64,321` bytes; SHA-256 `c3c60bfa2f68a12bc87d7c4ebeaa749c85c131bfc8a650cf97fcfe722883b4d7`; manifest integrity SHA-256 `e92fa1ed814f3ce79a7b502536a42ec9eb390a9a7f8ae87ab550daebe65d599e`). It is not a release or accepted representation.
- Decision: **change strategy**. E5M2 scalar-scale precision is not exact, so do not train or extend it. No model process is active; the next hierarchy candidate must alter scale sharing or encoding and establish independent red-to-green parity before exhaustive evaluation.

## Rejected packed INT2 two-value shared-FP16-scale candidate

`complete-int2-two-value-shared-fp16-scales-research`

- A distinct hierarchy candidate quantized immutable `artifacts-fp32.pt` into low-bit-first packed INT2 codes with one shared `float16` scale for each contiguous two-value group. It has `12,364` scales rather than the scalar proof's `24,726`; accepted F32, INT4, INT3, scalar INT2 proof, and prior FP16/FP8 candidates remain unchanged.
- TDD evidence: `tests/test_packed_int2_group2_fp16_scales.py` failed because the FP16 group-2 module was absent, then passed after the minimal independent runtime implementation. `tests/test_packed_int2_group2_fp16_preflight.py` likewise failed for its absent runner, then passed. Full suite: **155 passed**.
- The packed artifact integrity gate rejected a one-bit packed-payload flip. Exhaustive independent runtime evaluation failed the exact policy gate with **29,689 / 294,778** legal-policy misses. Immutable rejected report: `artifacts/rejected/int2-packed-group2-fp16-scales-preflight-20260924/report.json`.
- Rejected artifact: `artifacts/rejected/int2-packed-group2-fp16-scales-preflight-20260924/crystal-9-int2-packed-group2-fp16-scales.pt` (`63,849` bytes; SHA-256 `bb03397f8f515613e1d0279647159dbcdec5e369594225271e5772f75b515d48`; manifest integrity SHA-256 `127c9e3ac80fdd26d7c2f012d0b7580d3f8688d7ae225ac36f05910df0855b4a`). It is not a release or accepted representation.
- Decision: **change strategy**. FP16 precision cannot make the two-value shared-scale hierarchy exact, so do not train or extend it. No model process is active; a future hierarchy candidate must alter group scale sharing or encoding and establish independent red-to-green parity before exhaustive evaluation.

## Accepted packed scalar INT2 lossless FP8-scale transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-zlib`

- A distinct representation candidate retains the immutable scalar-FP8 INT2 layout and its genuine low-bit-first packed INT2 codes, but transports the complete FP8 scale stream as a losslessly zlib-compressed payload. Accepted F32, INT4, INT3, the scalar INT2 proof, and every earlier INT2 research artifact remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_zlib.py` first failed because the compressed-runtime module was absent, then passed after its minimal implementation. The payload-integrity test was run red with the integrity check removed, then green after it was restored. Full suite: **157 passed**.
- Exhaustive independent runtime gate passed with **0 /294,778** legal-policy misses. The integrity gate rejects a one-bit compressed-scale payload flip before decompression. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-zlib-preflight-20260924/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-zlib-preflight-20260924/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-zlib.pt` (`52,443` bytes; SHA-256 `8950a00cdd10e0542f4f332c652c2ef1783c9ddc9dbdbb85bc926feeb82e6d93`; manifest integrity SHA-256 `a310e0b39359862b32e54ca453cf09ef2942e7ffa2470dbabd2680858f04626d`). Its 24,726 raw FP8 scale bytes compress losslessly to 18,011 bytes.
- This reduces the prior exact scalar-FP8 artifact from 64,525 to 52,443 bytes but remains a scalar-scale hierarchy research artifact, not a claimed compact/deployable INT2 release: no release manifest or distribution gates exist.
- Decision: **advance**. No training process is active. The next representation candidate must be independently parity-tested and may only make a further lossless transport/packing improvement or a distinct hierarchy change; it must preserve this accepted artifact and pass exhaustive runtime plus integrity gates.

## Accepted packed scalar INT2 lossless FP8-scale LZMA transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma`

- A distinct lossless transport candidate preserves the immutable scalar-FP8 INT2 codes and scale values but encodes its FP8 scale stream with LZMA instead of zlib. All accepted F32, INT4, INT3, scalar INT2, and zlib INT2 artifacts are unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma.py` failed for an absent module and passed after implementation; its integrity check was red with validation removed and green after restoration.
- Exhaustive independent runtime gate passed with **0 /294,778** legal-policy misses. A compressed-payload bit flip is rejected before decompression. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-preflight-20260924/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-preflight-20260924/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma.pt` (`51,675` bytes; SHA-256 `7885932804257775709d90aa40972da15c6df57571f00ab29a122bc8ca267489`; manifest integrity SHA-256 `b71b4ee1c613ffe81dd2614458e36665e5f3a5a2e001893298c148c903ce843d`). The 24,726 raw FP8 scale bytes encode to 17,552 LZMA bytes.
- This improves the exact zlib candidate by 768 bytes, but remains a scalar-scale hierarchy research artifact rather than a compact/deployable INT2 release; release manifest and distribution integrity gates remain absent.
- Decision: **advance**. No training process is active. The next representation candidate must be independently parity-tested and either find a further lossless transport/packing improvement or define a separate hierarchy; it must preserve all accepted artifacts and pass exhaustive runtime plus integrity gates.

## Rejected packed scalar INT2 lossless FP8-scale BZIP2 transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-bzip2`

- A distinct lossless BZIP2 transport was tested against the immutable F32 source and the accepted scalar-FP8 INT2 layout; it preserves packed codes and FP8 scale bytes. Earlier accepted artifacts remain unchanged.
- TDD evidence: the BZIP2 runtime/integrity test was red for the absent module, then green; the preflight report builder test was red for the absent runner, then green. Full suite: **162 passed**.
- The independent packed-runtime gate passed exactly: **0 /294,778** legal-policy misses. A compressed-payload bit flip is rejected before decompression. Candidate report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-bzip2-preflight-20260924/report.json`.
- It is rejected as a compression-improvement candidate: its 24,726 raw FP8 scale bytes compressed to **18,537** bytes and its artifact is **53,265** bytes, both worse than accepted LZMA transport (**17,552** scale bytes; **51,675**-byte artifact). It is not a release or deployment claim.
- Decision: **change strategy**. No model process is active. Do not extend or promote BZIP2; retain accepted LZMA transport. Any successor needs its own parity/integrity gate and must provide a real packing/transport improvement or a distinct hierarchy.

## Rejected packed scalar INT2 raw-LZMA FP8-scale transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-raw`

- A distinct lossless raw-LZMA2 transport was evaluated against immutable `artifacts-fp32.pt`; it preserves the scalar-FP8 scale bytes and genuine low-bit-first packed INT2 codes. Accepted F32, INT4, INT3, scalar INT2, and LZMA artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma_raw.py` failed for the absent transport module, then passed after the minimal runtime implementation. It proves fake materialization equivalence and rejects a compressed-payload bit flip before decompression.
- The independent exhaustive runtime gate passed: **0 /294,778** legal-policy misses. Immutable candidate report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-raw-preflight-20260924/report.json`.
- It is rejected as a transport improvement: raw-LZMA reduced the scale payload from accepted LZMA's **17,552** to **17,495** bytes, but expanded the serialized artifact from **51,675** to **51,891** bytes. Candidate artifact SHA-256: `d9081f376f4c0066fe6675a30ebf8014d4ffcdd57ece9952e981c71e1a3c81ec`; manifest integrity SHA-256: `e627531e631bcf8c891afddc5c212dba30077b90e646c07f38dc0f1f04d50fe2`.
- Decision: **change strategy**. No model process is active. Preserve the accepted LZMA transport; do not promote or extend raw-LZMA. A successor must establish fresh parity/integrity coverage and improve the complete serialized artifact or use a distinct hierarchy.

## Rejected packed scalar INT2 gzip FP8-scale transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-gzip`

- A distinct lossless gzip transport was evaluated against immutable `artifacts-fp32.pt`; it preserves scalar-FP8 scale bytes and genuine low-bit-first packed INT2 codes. Accepted F32, INT4, INT3, scalar INT2, zlib, and LZMA artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_gzip.py` was red because its transport module was absent, then green after the minimal runtime implementation. `tests/test_verify_packed_int2_fp8_gzip.py` was red because its report runner was absent, then green. Full suite: **167 passed**.
- The independent exhaustive runtime gate passed: **0 /294,778** legal-policy misses. A compressed-payload bit flip is rejected before decompression. Immutable candidate report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-gzip-preflight-20260924/report.json`.
- It is rejected as a transport improvement: gzip stored the 24,726 raw FP8 scale bytes in **18,023** bytes and produced a **52,443**-byte artifact, versus accepted LZMA's **17,552** scale bytes and **51,675**-byte artifact. Candidate artifact SHA-256: `f447a0db5a545b4ae0b6b7336f01a5ffca785076c47b97a874dbc2c071c71b82`; manifest integrity SHA-256: `1bee4b49b02be7107ee629905aa811692650466202f95db2c3bbdfcc841cf239`.
- Decision: **change strategy**. No model process is active. Preserve the accepted LZMA transport; do not promote or extend gzip. A successor must establish fresh parity/integrity coverage and improve the complete serialized artifact or use a distinct hierarchy.

## Rejected packed scalar INT2 Zstandard FP8-scale transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-zstd`

- A distinct lossless Zstandard transport was evaluated against immutable `artifacts-fp32.pt`; it preserves scalar-FP8 scale bytes and genuine low-bit-first packed INT2 codes. Accepted F32, INT4, INT3, scalar INT2, zlib, and LZMA artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_zstd.py` was red because its transport module was absent, then green after the minimal runtime implementation. `tests/test_verify_packed_int2_fp8_zstd.py` was red because its report runner was absent, then green. Full suite: **170 passed**.
- The independent exhaustive runtime gate passed: **0 /294,778** legal-policy misses. A compressed-payload bit flip is rejected before decompression. Immutable candidate report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-zstd-preflight-20260924/report.json`.
- It is rejected as a transport improvement: Zstandard stored the 24,726 raw FP8 scale bytes in **17,730** bytes and produced a **51,867**-byte artifact, versus accepted LZMA's **17,552** scale bytes and **51,675**-byte artifact. Candidate artifact SHA-256: `3d87b984c3802291920586a529825dfbe53cc3f68429047b57c9f2a26347a064`; manifest integrity SHA-256: `f4b13f0e352fd814f8fb31bfb89cb6dd5af47066fef48948f67ebcc28d7db5ff`.
- Decision: **change strategy**. No model process is active. Preserve the accepted LZMA transport; do not promote or extend Zstandard. Further work requires a distinct hierarchy/packing design rather than another generic lossless codec.

## Accepted packed scalar INT2 compact-binary LZMA FP8-scale transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-compact`

- A distinct compact-container candidate retains the immutable scalar-FP8 INT2 codes and losslessly LZMA-encoded scale stream, but replaces Torch serialization with an integrity-bound length-delimited binary container. Accepted F32, INT4, INT3, scalar INT2 proof, and all earlier transport artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma_compact.py` first failed because the compact runtime/export module was absent, then passed after minimal implementation. `tests/test_verify_packed_int2_fp8_lzma_compact.py` likewise first failed because its report runner was absent, then passed. Full suite: **173 passed**.
- The independent exhaustive packed-runtime gate passed: **0 /294,778** legal-policy misses. A compact-payload bit flip is rejected before decode. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-compact-preflight-20260924/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-compact-preflight-20260924/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-compact.c9i2` (**30,215** bytes; SHA-256 `3eca731be4780da1ded9c042a77a9efec3114c9972a5cbc8afcb10324a955442`; payload integrity SHA-256 `3256e21273c06685e3687ea8bab6e72fbb5d51118e3cabd1b88afe08bb306b9c`). It preserves the same 24,726 raw FP8 scale bytes compressed losslessly to 17,552 bytes and reduces the accepted Torch/LZMA artifact from 51,675 to 30,215 bytes.
- This is an exact, compact-container hierarchy-quantization research artifact, not a deployable INT2 release: it still has one scale per scalar and has no release manifest or distribution integrity gate.
- Decision: **advance**. No model process is active. The next candidate must be independently parity/integrity tested and either improve the complete compact container or define a distinct scale hierarchy; it must preserve this accepted artifact.

## Accepted packed scalar INT2 canonical-binary LZMA FP8-scale transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-binary`

- A distinct canonical-binary container preserves the immutable scalar-FP8 INT2 codes and lossless LZMA scale stream, while replacing the JSON tensor-offset manifest with a fixed, architecture-validated tensor ordering. Accepted F32, INT4, INT3, scalar INT2, and all earlier transport artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma_binary.py` was red because the binary module was absent, then green; `tests/test_verify_packed_int2_fp8_lzma_binary.py` likewise was red for its absent report runner, then green. Full suite: **176 passed**.
- The independent exhaustive packed-runtime gate passed: **0 / 294,778** legal-policy misses. A one-bit compressed-scale payload flip is rejected before decode. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-preflight-20260924/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-preflight-20260924/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary.c9i2` (**23,807** bytes; SHA-256 `52eaea7575b63dd8f15ed82be208e8b6325233577c2723a9dc9d99bbe40b6a66`; payload integrity SHA-256 `e57fdc69227ebba0ef2e4ca0764bec0513b4e54939a99c4c38f0d48cce3bd802`). It retains 24,726 raw FP8 scale bytes compressed losslessly to 17,552 bytes and reduces the accepted JSON-compact LZMA artifact from 30,215 to 23,807 bytes.
- This is an exact compact-container hierarchy-quantization research artifact, not a deployable INT2 release: scalar scales remain one per parameter, and release-manifest/distribution gates are absent.
- Decision: **advance**. No model process is active. The next candidate must establish fresh parity/integrity coverage and either reduce the complete binary artifact further or define a distinct scale hierarchy; it must preserve this accepted artifact.

## Accepted packed scalar INT2 LZMA-scale/Zlib-code canonical-binary transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-binary-zlib-codes`

- A distinct lossless compact-container candidate preserves the immutable scalar-FP8 INT2 values but independently compresses its 2-bit low-bit-first code payload with Zlib level 9 while retaining the LZMA scale stream. Accepted F32, INT4, INT3, scalar INT2 proof, and all earlier transport artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma_binary_zlib_codes.py` failed for the absent runtime module, then passed after minimal implementation; `tests/test_verify_packed_int2_fp8_lzma_binary_zlib_codes.py` failed for the absent report runner, then passed. Full suite: **179 passed**.
- The independent exhaustive packed-runtime gate passed: **0 / 294,778** legal-policy misses. A one-bit payload flip is rejected before either stream is decoded. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-zlib-codes-preflight-20260924/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-zlib-codes-preflight-20260924/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-zlib-codes.c9i2` (**21,171** bytes; SHA-256 `96a8e24d68475456f9d8d1cdc7035ded61d8361feb32f6e00b40dba16e27fb64`; payload integrity SHA-256 `acc74cf67415e603acaea107ccb261853b1a7da97f4e39a5ffb18f60b46df962`). It losslessly reduces the 6,183-byte packed-code stream to 3,543 bytes and lowers the accepted binary/LZMA artifact from 23,807 to 21,171 bytes.
- This is an exact compact-container hierarchy-quantization research artifact, not a deployable INT2 release: scalar scales remain one per parameter and release-manifest/distribution gates remain absent.
- Decision: **advance**. No model process is active. The next candidate must establish fresh parity/integrity coverage and either improve the complete binary artifact further or define a distinct scale hierarchy; it must preserve this accepted artifact.

## Rejected packed scalar INT2 LZMA-code canonical-binary transport screen

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-binary-lzma-codes`

- A bounded lossless-code-transport screen retained the accepted scalar-FP8 scale layout and LZMA scale stream, changing only the 6,183-byte packed-code stream from accepted Zlib-9 to LZMA-9. F32, INT4, INT3, the scalar INT2 proof, and all accepted INT2 artifacts are unchanged.
- TDD evidence: a fresh candidate test was red for the absent LZMA-code module. The minimal implementation proved materialization equivalence and payload-integrity rejection, but failed its required complete-artifact-improvement assertion and was removed rather than retained as production code.
- From immutable `artifacts-fp32.pt`, LZMA produced a 3,852-byte code stream and a 21,480-byte container, versus accepted Zlib's 3,543-byte code stream and 21,171-byte container. The candidate is 309 bytes larger in both measures.
- Immutable rejection report: `artifacts/rejected/int2-packed-scalar-fp8-lzma-binary-lzma-codes-20260924/report.json`. No exhaustive policy evaluation was run: the candidate cannot improve the accepted artifact on its primary representation metric.
- Decision: **change strategy**. No model process is active. Do not revisit generic LZMA code compression; a successor must use a distinct hierarchy/packing design and establish fresh parity/integrity coverage before exhaustive evaluation.

## Accepted packed scalar INT2 bitplane-Zlib-code canonical-binary transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-binary-bitplane-zlib-codes`

- A distinct code-packing candidate preserves the immutable full scalar FP8-scale INT2 layout and LZMA scale stream, but transposes the low-bit-first two-bit codes into separate low/high bitplanes before Zlib-9 compression. Accepted F32, INT4, INT3, scalar INT2 proof, and all earlier INT2 artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes.py` and `tests/test_verify_packed_int2_fp8_lzma_binary_bitplane_zlib_codes.py` each failed first for their absent modules, then passed after minimal implementation. Full suite: **182 passed**.
- The independent exhaustive packed-runtime gate passed: **0 / 294,778** legal-policy misses. The payload integrity gate rejects a one-bit payload flip before either compressed stream is decoded. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-bitplane-zlib-codes-preflight-20260925/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-bitplane-zlib-codes-preflight-20260925/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-bitplane-zlib-codes.c9i2` (**20,811** bytes; SHA-256 `5eeca21ace9671200360e8e6cdb7b45e15f1cd98448912e70e4fa5a3a4766ff2`; payload integrity SHA-256 `624117d4c9190cbc5e00d56d3bd9f8cd3538469d4e3f69a6321bb40c62b88eb6`). Bitplane packing reduces the Zlib code stream from 3,543 to 3,183 bytes and the accepted container from 21,171 to 20,811 bytes.
- This is an exact compact-container hierarchy-quantization research artifact, not a deployable INT2 release: it retains one FP8 scale per scalar and release-manifest/distribution gates remain absent.
- Decision: **advance**. No model process is active. The next candidate must establish fresh parity/integrity coverage for a distinct complete-container packing improvement or a different scale hierarchy; it must preserve this accepted artifact.

## Accepted packed scalar INT2 permutation-selected-bitplane-Zlib-code canonical-binary transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-binary-permuted-bitplane-zlib-codes`

- A distinct code-packing candidate preserves the immutable full scalar FP8-scale INT2 layout and LZMA scale stream, selecting the smallest of all 24 reversible two-bit code permutations before bitplane transposition and Zlib-9 compression. Accepted F32, INT4, INT3, scalar INT2 proof, and all earlier INT2 artifacts remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes.py` failed first because the runtime module was absent, then passed after minimal implementation; `tests/test_verify_packed_int2_fp8_lzma_binary_permuted_bitplane_zlib_codes.py` likewise failed for its absent report runner, then passed. Targeted tests: **3 passed**.
- The independent exhaustive packed-runtime gate passed: **0 / 294,778** legal-policy misses. The payload integrity gate rejects a one-bit payload flip before either compressed stream is decoded. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-permuted-bitplane-zlib-codes-preflight-20260925/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-permuted-bitplane-zlib-codes-preflight-20260925/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-permuted-bitplane-zlib-codes.c9i2` (**20,806** bytes; SHA-256 `6564a958b317e7077a22799d69d10a11989cf149ab9ed6f3ee8f830e87cd2f96`; payload integrity SHA-256 `3dce88d8dd5dc1a9a7727fdc2ccbcccea9ee9b37ea9ca42879cbf3ea3a69f18f`). The selected reversible permutation `[0, 3, 1, 2]` reduces the bitplane Zlib code stream from 3,183 to 3,178 bytes and the accepted container from 20,811 to 20,806 bytes.
- This is an exact compact-container hierarchy-quantization research artifact, not a deployable INT2 release: it retains one FP8 scale per scalar and release-manifest/distribution gates remain absent.
- Decision: **advance**. No model process is active. The next candidate must establish fresh parity/integrity coverage for a materially distinct complete-container packing improvement or a different scale hierarchy; it must preserve this accepted artifact.

## Rejected INT2 static-arithmetic-code transport screen

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-binary-arithmetic-codes`

- A distinct static-arithmetic coding screen was attempted for the low-bit-first scalar INT2 code symbols while retaining the immutable FP8 scale stream and LZMA scale transport.
- TDD evidence: a fresh runtime/materialization and complete-container-improvement test was red for the absent module. Minimal implementation established local materialization and integrity behavior, but the required size assertion remained red: **18,658** bytes versus **18,543** bytes for the accepted permutation-bitplane/Zlib baseline in the fixed test fixture.
- The candidate was therefore removed rather than retained as production code. It has no immutable artifact or exhaustive gate because it failed the primary complete-container-improvement gate before policy evaluation.
- Decision: **change strategy**. Preserve the accepted 20,806-byte exhaustive artifact. No model process is active; the next candidate must use a materially different hierarchy or packing design, not another generic entropy-code variation.

## Rejected INT2 tensor-local code-permutation transport screen

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-binary-tensor-permuted-bitplane-zlib-codes`

- A bounded packing screen retained the accepted scalar-FP8 scale stream and bitplane/Zlib code transport, but selected a reversible two-bit code permutation separately for each canonical model tensor rather than once globally.
- TDD evidence: the fresh runtime/materialization and complete-container-improvement test was red for the absent module. The minimal implementation then passed materialization and payload-bit-flip integrity assertions, but failed its required complete-container-improvement assertion.
- On the fixed `TinyMoEPolicy(vocab_size=13)` test fixture, the tensor-local table produced a **18,629**-byte container versus **18,534** bytes for the accepted global-permutation baseline: 95 bytes larger. The per-tensor permutation table cost exceeded its compression effect.
- The candidate module and test were removed rather than retained as production code. No immutable artifact or exhaustive policy evaluation was created because it failed the primary representation gate.
- Decision: **change strategy**. The accepted 20,806-byte exhaustive artifact remains immutable; no model process is active. A successor requires a materially different scale hierarchy or complete-container design, not another permutation/entropy variation.

## Rejected INT2 lossless FP8-scale palette hierarchy screen

`complete-scalar-group-int2-packed-fp8-scale-palette-hierarchy-screen`

- A distinct lossless scale-hierarchy screen was measured directly from immutable `artifacts-fp32.pt`: the scalar-FP8 scale stream contains 24,726 bytes but only 78 distinct FP8 values. The candidate would retain the exact 78-byte palette and replace each scale with a fixed 7-bit palette index, preserving every dequantized scalar exactly.
- The fixed-width index stream requires 21,636 bytes before its palette, versus the accepted canonical LZMA scale payload's 17,552 bytes. LZMA-9 compresses the index stream to 21,208 bytes, still larger than the accepted scale payload before container metadata.
- It is rejected at the primary complete-representation screen; no runtime/module, artifact, or exhaustive policy run was created because the exact hierarchy cannot improve the accepted 20,806-byte container. Accepted F32, INT4, INT3, scalar INT2 proof, and all accepted INT2 transports remain unchanged.
- Decision: **change strategy**. No model process is active. Further work must be a materially different exact scale hierarchy or complete-container design; do not run another generic code/scale entropy variation.

## Rejected INT2 FP8-scale delta-stream screen

`complete-scalar-group-int2-packed-fp8-scale-delta-stream-screen`

- A bounded, reversible scale-order screen measured first-order modular-delta and XOR-delta transforms of the immutable 24,726-byte scalar-FP8 scale stream before LZMA-9. Both retain the exact FP8 scale bytes after inversion and leave the accepted scalar INT2 codes unchanged.
- Neither transform improves the accepted 17,552-byte canonical LZMA scale payload: modular delta produced 18,948 bytes and XOR delta produced 18,824 bytes. No runtime, artifact, or exhaustive policy evaluation was created because both fail the complete-representation primary metric.
- Decision: **change strategy**. No model process is active. This eliminates adjacent-scale predictive transforms; the next candidate must be a materially different exact hierarchy or complete-container design, not another generic entropy transform.

## Rejected INT2 FP8-scale canonical-Huffman hierarchy screen

`complete-scalar-group-int2-packed-fp8-scale-canonical-huffman-hierarchy-screen`

- A bounded exact palette-hierarchy screen measured the immutable 24,726-byte scalar-FP8 scale stream's canonical Huffman lower bound. The stream has 78 distinct FP8 values and a Shannon entropy of 5.673440918994552 bits per scale; its optimal Huffman stream requires 140,941 bits (17,618 bytes), before the required palette and canonical code-length metadata.
- The accepted canonical LZMA scale payload is 17,552 bytes. Even the metadata-free optimal Huffman stream is 66 bytes larger, so this exact hierarchy cannot improve the accepted 20,806-byte complete container. No runtime, artifact, or exhaustive policy evaluation was created; accepted F32, INT4, INT3, scalar INT2 proof, and INT2 transport artifacts remain unchanged.
- Decision: **change strategy**. No model process is active. Canonical symbol-code scale hierarchies are now screened out; a successor must introduce a materially different exact hierarchy or complete-container design, not another generic entropy transform.

## Rejected INT2 code-conditioned FP8-scale ordering screen

`complete-scalar-group-int2-packed-fp8-scale-code-conditioned-ordering-screen`

- A distinct lossless hierarchy screen used the immutable `artifacts-fp32.pt` scalar-FP8 INT2 values. It stably partitioned the 24,726 FP8 scales by their corresponding decoded INT2 code before LZMA-9, retaining the original scale order within every code partition; decoder reconstruction would use the already-integrity-bound code stream to invert the partition exactly.
- The code inventory is `[35, 12,162, 0, 12,529]` for symbols `[0, 1, 2, 3]`. All 24 reversible code-partition orders were measured. The best compressed scale payload is **17,588** bytes (order `[0, 1, 2, 3]`), versus **17,552** bytes for the accepted canonical scale order; the worst is 17,592 bytes.
- It is rejected at the primary complete-representation screen: even the best conditioned scale stream is 36 bytes larger before any required ordering metadata or container changes. No runtime/module, artifact, or exhaustive policy evaluation was created. Accepted F32, INT4, INT3, scalar INT2 proof, and all accepted INT2 transport artifacts remain unchanged.
- Decision: **change strategy**. No model process is active. Code-conditioned scale ordering does not improve this exact scalar hierarchy; the successor must use a materially different exact scale hierarchy or complete-container design, not another generic entropy transform.

## Accepted packed scalar INT2 fixed-binary LZMA-scale/permuted-bitplane-Zlib-code transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-fixed-binary-permuted-bitplane-zlib-codes`

- A distinct fixed-architecture container preserves the immutable scalar-FP8 INT2 layout, LZMA scale stream, selected reversible code permutation, and bitplane/Zlib code stream while omitting architecture fields that the fixed `TinyMoEPolicy` runtime derives and validates. Accepted F32, INT4, INT3, scalar INT2 proof, and every prior INT2 artifact remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma_fixed_binary_permuted_bitplane_zlib_codes.py` was red for the absent runtime module, then green after minimal implementation. `tests/test_verify_packed_int2_fp8_lzma_fixed_binary_permuted_bitplane_zlib_codes.py` likewise was red for the absent report runner, then green. Full suite: **188 passed**.
- The independent exhaustive packed-runtime gate passed with **0 / 294,778** legal-policy misses. The payload-integrity gate rejects a one-bit payload flip before either compressed stream is decoded. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-binary-permuted-bitplane-zlib-codes-preflight-20260925/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-binary-permuted-bitplane-zlib-codes-preflight-20260925/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-binary-permuted-bitplane-zlib-codes.c9i2` (**20,790** bytes; SHA-256 `5c97b9cda72d1f0cf73a927e94919d356e052611de02cdbf0a82db9ade472113`; payload integrity SHA-256 `6b3b09f45e279ce9c7d1822e784c272c5f5f20ff55bba665cfe17e44d9d26c44`). It retains the same 3,178-byte code and 17,552-byte scale streams while reducing the accepted architecture-described container from 20,806 to 20,790 bytes.
- This is an exact compact-container hierarchy-quantization research artifact, not a deployable INT2 release: it retains one FP8 scale per scalar and release-manifest/distribution gates remain absent.
- Decision: **advance**. No model process is active. The next candidate must establish fresh parity/integrity coverage for a materially distinct exact scale hierarchy or complete-container design; it must preserve this accepted artifact.

## Accepted packed scalar INT2 fixed-stream LZMA-scale/permuted-bitplane-Zlib-code transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-stream-binary-permuted-bitplane-zlib-codes`

- A distinct fixed-stream container preserves the immutable scalar-FP8 INT2 layout, LZMA scale stream, selected reversible code permutation, and bitplane/Zlib code stream while deriving fixed Crystal-9 tensor sizes and finding the scale/code boundary from the self-terminating LZMA stream. Accepted F32, INT4, INT3, scalar INT2 proof, and every prior INT2 artifact remain unchanged.
- TDD evidence: `tests/test_packed_int2_fp8_scale_lzma_stream_binary_permuted_bitplane_zlib_codes.py` was red for the absent runtime module, then green after minimal implementation; `tests/test_verify_packed_int2_fp8_lzma_stream_binary_permuted_bitplane_zlib_codes.py` was red for the absent report runner, then green. Full suite: **191 passed**.
- The independent exhaustive packed-runtime gate passed with **0 / 294,778** legal-policy misses. The payload-integrity gate rejects a one-bit payload flip before decode. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-stream-binary-permuted-bitplane-zlib-codes-preflight-20260925/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-stream-binary-permuted-bitplane-zlib-codes-preflight-20260925/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-stream-binary-permuted-bitplane-zlib-codes.c9i2` (**20,770** bytes; SHA-256 `afeded554447f0d8802da57659584759033071f73a465abec96239b8c828a4d7`; payload integrity SHA-256 `653a0c0627fa648c1c79c6291ac5e445056eba69c9e0b78fb42d689c551539f6`). It retains the same 3,178-byte code and 17,552-byte scale streams while reducing the fixed-binary 20,790-byte artifact by 20 bytes.
- This is an exact compact-container hierarchy-quantization research artifact, not a deployable INT2 release: it retains one FP8 scale per scalar and release-manifest/distribution gates remain absent.
- Decision: **advance**. No model process is active. The next candidate must establish fresh parity/integrity coverage for a materially distinct exact scale hierarchy or complete-container design; it must preserve this accepted artifact.

## Accepted packed scalar INT2 fixed-permutation-stream LZMA-scale/bitplane-Zlib-code transport candidate

`complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes`

- A distinct fixed-permutation container preserves the immutable scalar-FP8 INT2 layout, LZMA scale stream, and bitplane/Zlib code stream, but fixes the accepted source-specific reversible code permutation `[0, 3, 1, 2]` in the format rather than serializing its one-byte selector. Accepted F32, INT4, INT3, scalar INT2 proof, and all prior INT2 artifacts remain unchanged.
- TDD evidence: runtime/materialization and report-builder tests were red for absent modules, then green after minimal implementations. Targeted tests: **3 passed**.
- The independent exhaustive packed-runtime gate passed with **0 / 294,778** legal-policy misses. The payload-integrity gate rejects a one-bit payload flip before decode. Immutable report: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes-preflight-20260925/report.json`.
- Artifact: `artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes-preflight-20260925/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes.c9i2` (**20,769** bytes; SHA-256 `95904590b98741c883bd48c481b39eabe0e05c9615c1f4e56a6d0535252c849f`; payload integrity SHA-256 `58edbad7e20aad586e930c9f2e4a3d3f384318d07aae5b5fd48b74e5c976bc98`). It retains the same 3,178-byte code and 17,552-byte scale streams while reducing the accepted stream-binary artifact from 20,770 to 20,769 bytes.
- This is an exact compact-container hierarchy-quantization research artifact, not a deployable INT2 release: it retains one FP8 scale per scalar and release-manifest/distribution gates remain absent.
- Decision: **change strategy**. No model process is active. The selector-elision result is a one-byte format-specific improvement; do not pursue another header/minor-entropy variant. The next candidate must be a materially different exact scale hierarchy or complete-container design and must preserve this accepted artifact.

## Rejected INT2 joint scale/code LZMA hierarchy screen

`complete-scalar-group-int2-packed-fp8-joint-scale-code-lzma-screen`

- A bounded, lossless joint-stream hierarchy screen used the immutable FP8 scalar-scale INT2 values from `artifacts-fp32.pt`. It compared LZMA-9 of (a) interleaved FP8-scale/unpacked-INT2-symbol bytes and (b) the two reversible whole-stream concatenation orders against the accepted separate streams.
- The 24,726-value candidate's best reversible joint stream was symbols-then-scales at **21,256** bytes; scales-then-symbols was 21,260 and interleaving was 23,500. The accepted fixed-permutation transport's separate LZMA-scale plus bitplane-Zlib-code streams total **20,730** bytes and its complete integrity-bound artifact is **20,769** bytes.
- The best joint-stream lower-bound container is 21,295 bytes before extra decoder metadata: **526 bytes larger** than the accepted artifact. Immutable rejection report: `artifacts/rejected/int2-packed-scalar-fp8-joint-scale-code-lzma-screen-20260925/report.json`.
- No runtime/module, model artifact, or exhaustive policy evaluation was created because this distinct hierarchy failed the primary complete-representation metric. All accepted F32, INT4, INT3, scalar INT2 proof, and INT2 transport artifacts remain unchanged.
- Decision: **change strategy**. No model process is active. Joint symbol/scale LZMA is screened out; successor research must use a materially different exact hierarchy rather than another generic codec or header variation.

## Rejected INT2 tensor-local LZMA scale hierarchy screen

`complete-scalar-group-int2-packed-fp8-tensor-local-lzma-scale-hierarchy-screen`

- A bounded, lossless scale-hierarchy screen split the immutable `artifacts-fp32.pt` scalar-FP8 scale bytes at each of the fixed `TinyMoEPolicy` tensor boundaries, then compressed each scale segment independently with LZMA-9. The fixed runtime can derive every boundary, so no tensor table is needed to decode the streams sequentially.
- The 24,726 exact FP8 scales form 48 tensor streams. Their aggregate compressed size is **21,740** bytes, versus **17,552** bytes for the accepted single canonical LZMA scale stream: **4,188 bytes larger**. Holding the same 3,178-byte integrity-bound code stream and 39-byte fixed container overhead, the candidate lower-bound container is **24,957** bytes, **4,188 bytes** larger than the accepted 20,769-byte artifact.
- It is rejected before runtime/artifact/exhaustive policy evaluation because the exact hierarchy fails the primary complete-representation metric even without additional version metadata. Immutable rejection report: `artifacts/rejected/int2-packed-scalar-fp8-tensor-local-lzma-scale-hierarchy-screen-20260925/report.json`.
- Decision: **change strategy**. No model process is active. Tensor-local reset points are screened out; successor work must be a materially different exact hierarchy rather than another stream-order or generic-codec variation.

## Checksum-bound exact INT2 staged-research distribution

`crystal-9-int2-staged-research-distribution-v1`

- The accepted fixed-permutation stream-binary scalar-FP8 artifact is now distributed with its immutable exhaustive acceptance report, fixed runtime-source closure, design, release manifest, README, and a SHA-256 manifest. The distribution explicitly remains a **staged research artifact; not a deployable INT2 release**: it retains one FP8 scale for each of 24,726 parameter scalars.
- TDD evidence: `tests/test_build_int2_research_distribution.py` was red because the distribution builder was absent, then green after its minimal implementation. Full suite: **195 passed**.
- The source acceptance remains exactly **0 / 294,778** legal-policy misses in the independent packed runtime. `releases/int2-staged-research-v1/SHA256SUMS` validates all ten distributed payload and provenance files.
- Decision: **advance**. Behavioral and distribution-integrity gates for this exact scalar-scale research representation are complete. No model process is active. Compactness remains blocked on a materially different exact scale hierarchy; do not revisit generic entropy, header, stream-order, or tensor-reset variants already rejected.

## Rejected INT2 FP8-scale bitplane/LZMA hierarchy screen

`complete-scalar-group-int2-packed-fp8-scale-bitplane-lzma-screen`

- A bounded, lossless scale-layout screen transposed the 24,726 immutable `float8_e4m3fn` scalar-scale bytes into eight packed bitplanes before LZMA-9, retaining the accepted fixed-permutation bitplane/Zlib code stream unchanged. Accepted F32, INT4, INT3, scalar INT2 proof, and all accepted INT2 transport artifacts remain unchanged.
- The transform expands the raw scale stream from 24,726 to 24,728 bytes and produces a **19,056-byte** LZMA scale payload, versus **17,552 bytes** for the accepted canonical stream. With the unchanged 3,178-byte code stream and 39-byte container overhead, its lower-bound artifact is **22,273 bytes**, **1,504 bytes larger** than the accepted 20,769-byte artifact.
- It is rejected before runtime/module, artifact, or exhaustive policy evaluation because the exact representation fails the primary complete-artifact metric. Immutable rejection report: `artifacts/rejected/int2-packed-scalar-fp8-scale-bitplane-lzma-screen-20260925/report.json`.
- Decision: **change strategy**. No model process is active. The checksum-bound staged-research distribution remains valid; successor research must define a materially different exact scale hierarchy rather than a further scale-stream entropy transform.

## Rejected INT2 FP8-scale component-stream/LZMA hierarchy screen

`complete-scalar-group-int2-packed-fp8-scale-component-stream-lzma-screen`

- A materially distinct, lossless scale-hierarchy screen split every immutable `float8_e4m3fn` scalar-scale byte into exponent and mantissa component streams before LZMA-9, retaining the accepted fixed-permutation bitplane/Zlib code stream unchanged. Accepted F32, INT4, INT3, scalar INT2 proof, and all accepted INT2 transport artifacts remain unchanged.
- The two component streams contain **49,452** bytes before compression and produce an **18,520-byte** LZMA payload, versus **17,552 bytes** for the accepted canonical 24,726-byte scale stream. With the unchanged 3,178-byte code stream and 39-byte container overhead, its lower-bound artifact is **21,737 bytes**, **968 bytes** larger than the accepted 20,769-byte artifact.
- It is rejected before runtime/module, artifact, or exhaustive policy evaluation because the exact representation fails the primary complete-artifact metric. Immutable rejection report: `artifacts/rejected/int2-packed-scalar-fp8-scale-component-stream-lzma-screen-20260925/report.json`.
- Decision: **change strategy**. No model process is active. Component decomposition does not improve this exact scalar hierarchy; a successor must be a materially different exact scale hierarchy, not another generic stream transform.
