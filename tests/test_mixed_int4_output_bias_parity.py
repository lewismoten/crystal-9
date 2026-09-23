import torch

from crystal9 import (
    TinyMoEPolicy,
    materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias_expert_biases_output_bias,
)


def test_output_bias_int4_forward_matches_materialized_runtime():
    torch.manual_seed(54)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    tokens = torch.tensor([[1, 6, 2, 8, 0, 0, 0, 0, 0]])
    groups, bias_groups = frozenset({"q", "k", "v", "out"}), frozenset({"in", "out"})
    expected = source.forward_mixed_int4_input_attention_groups(
        tokens, groups, attention_bias_groups=bias_groups, quantize_router_weight=True,
        quantize_router_bias=True, quantize_expert_biases=True, quantize_output_bias=True,
    )
    actual = materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias_expert_biases_output_bias(source)(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
