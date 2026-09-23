import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_full_parameters


def test_full_parameter_int4_forward_matches_materialized_runtime():
    torch.manual_seed(55)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    tokens = torch.tensor([[1, 6, 2, 8, 0, 0, 0, 0, 0]])
    groups, bias_groups = frozenset({"q", "k", "v", "out"}), frozenset({"in", "out"})
    expected = source.forward_mixed_int4_input_attention_groups(
        tokens, groups, attention_bias_groups=bias_groups, quantize_router_weight=True,
        quantize_router_bias=True, quantize_expert_biases=True, quantize_output_bias=True,
        quantize_norm=True,
    )
    actual = materialize_mixed_int4_full_parameters(source)(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
