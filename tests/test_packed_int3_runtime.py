import torch

from crystal9 import (
    TinyMoEPolicy,
    materialize_mixed_int3_scalar_input_attention_q_k_group2_v_group2_out_group1_input_bias_group1_output_bias_group1_norm_weight_group1_norm_bias_group1,
)
from packed_int3 import PackedInt3Policy, export_packed_int3


def test_packed_int3_runtime_matches_complete_materialized_layout(tmp_path):
    torch.manual_seed(101)
    source = TinyMoEPolicy(vocab_size=13).eval()
    artifact_path = tmp_path / "crystal-9-int3.pt"

    manifest = export_packed_int3(source, artifact_path)
    runtime = PackedInt3Policy.load(artifact_path).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    expected = materialize_mixed_int3_scalar_input_attention_q_k_group2_v_group2_out_group1_input_bias_group1_output_bias_group1_norm_weight_group1_norm_bias_group1(source)(token_ids)

    torch.testing.assert_close(runtime(token_ids), expected)
    assert manifest["format"] == "crystal-9-packed-int3-v1"
    assert manifest["parameter_values"] == sum(parameter.numel() for parameter in source.parameters())
