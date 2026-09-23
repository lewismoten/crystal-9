import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_input_attention_q_v_out_k


def test_key_int4_forward_matches_materialized_runtime():
    torch.manual_seed(29)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    tokens = torch.tensor([[1, 6, 2, 8, 0, 0, 0, 0, 0]])

    expected = source.forward_mixed_int4_input_attention_groups(tokens, frozenset({"q", "k", "v", "out"}))
    actual = materialize_mixed_int4_input_attention_q_v_out_k(source)(tokens)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
