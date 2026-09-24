import torch

from crystal9 import (
    TinyMoEPolicy,
    materialize_mixed_int2_scalar_suffix_output_bias_router_weight_bias_expert_biases_position_embedding_attention_q_group1_k_group1_v_group1_out_group1_in_bias_group1_out_bias_group1_norm_weight_group1_norm_bias_group1,
)
from packed_int2 import PackedInt2Policy, evaluate_packed, export_packed_int2


def test_packed_int2_runtime_matches_full_scalar_materialized_layout(tmp_path):
    torch.manual_seed(20260977)
    source = TinyMoEPolicy(vocab_size=13).eval()
    artifact_path = tmp_path / "crystal-9-int2.pt"

    manifest = export_packed_int2(source, artifact_path)
    runtime = PackedInt2Policy.load(artifact_path).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    expected = materialize_mixed_int2_scalar_suffix_output_bias_router_weight_bias_expert_biases_position_embedding_attention_q_group1_k_group1_v_group1_out_group1_in_bias_group1_out_bias_group1_norm_weight_group1_norm_bias_group1(source)(token_ids)

    torch.testing.assert_close(runtime(token_ids), expected)
    assert manifest["format"] == "crystal-9-packed-int2-v1"
    assert manifest["parameter_values"] == sum(parameter.numel() for parameter in source.parameters())


def test_packed_int2_evaluation_counts_policy_misses_for_supplied_histories():
    class ConstantPolicy:
        def __call__(self, token_ids):
            return torch.tensor([[0.0, 1.0]]).repeat(token_ids.shape[0], 1)

    class Tokenizer:
        def encode_history(self, history):
            return [1] * (len(history) + 1)

        def decode_id(self, token_id):
            return "b" if token_id == 1 else "a"

    result = evaluate_packed(
        ConstantPolicy(), Tokenizer(), torch.device("cpu"), ["", "a"], lambda history: "a" if history == "" else "b"
    )

    assert result == {"legal_histories": 2, "policy_misses": 1}
