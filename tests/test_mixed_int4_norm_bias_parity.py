import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_norm_bias


def test_norm_bias_int4_forward_matches_materialized_runtime():
    torch.manual_seed(56)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    tokens = torch.tensor([[1, 6, 2, 8, 0, 0, 0, 0, 0]])
    groups, biases = frozenset({"q", "k", "v", "out"}), frozenset({"in", "out"})
    expected = source.forward_mixed_int4_input_attention_groups(
        tokens, groups, attention_bias_groups=biases, quantize_router_weight=True,
        quantize_router_bias=True, quantize_expert_biases=True, quantize_output_bias=True,
        norm_int4_groups=frozenset({"bias"}),
    )
    torch.testing.assert_close(materialize_mixed_int4_norm_bias(source)(tokens), expected, rtol=0, atol=1e-6)
