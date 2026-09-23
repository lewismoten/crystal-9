import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_full_parameters_grouped_norm_weight


def test_grouped_norm_weight_int4_forward_matches_materialized_runtime():
    torch.manual_seed(58)
    source = TinyMoEPolicy(vocab_size=13).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    groups = frozenset({"q", "k", "v", "out"})
    bias_groups = frozenset({"in", "out"})
    norm_groups = frozenset({"weight", "bias"})
    with torch.no_grad():
        expected = source.forward_mixed_int4_input_attention_groups(
            token_ids, groups, attention_bias_groups=bias_groups,
            quantize_router_weight=True, quantize_router_bias=True,
            quantize_expert_biases=True, quantize_output_bias=True,
            norm_int4_groups=norm_groups, norm_weight_group_size=8,
        )
        actual = materialize_mixed_int4_full_parameters_grouped_norm_weight(source, 8)(token_ids)
    torch.testing.assert_close(actual, expected)
