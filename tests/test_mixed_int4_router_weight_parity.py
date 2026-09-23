import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight


def test_router_weight_int4_forward_matches_materialized_runtime():
    torch.manual_seed(41)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    tokens = torch.tensor([[1, 6, 2, 8, 0, 0, 0, 0, 0]])

    expected = source.forward_mixed_int4_input_attention_groups_with_router(
        tokens,
        frozenset({"q", "k", "v", "out"}),
        attention_bias_groups=frozenset({"out"}),
        quantize_router_weight=True,
    )
    actual = materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight(source)(tokens)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
