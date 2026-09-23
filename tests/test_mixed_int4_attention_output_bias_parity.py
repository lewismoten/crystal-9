import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_input_attention_q_v_out_k_output_bias


def test_attention_output_bias_int4_forward_matches_materialized_runtime():
    torch.manual_seed(37)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    tokens = torch.tensor([[1, 6, 2, 8, 0, 0, 0, 0, 0]])

    expected = source.forward_mixed_int4_input_attention_groups_with_biases(
        tokens, frozenset({"q", "k", "v", "out"}), attention_bias_groups=frozenset({"out"})
    )
    actual = materialize_mixed_int4_input_attention_q_v_out_k_output_bias(source)(tokens)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
