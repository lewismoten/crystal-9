import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_input_attention_q


def test_q_projection_int4_forward_matches_materialized_runtime():
    torch.manual_seed(17)
    source = TinyMoEPolicy(vocab_size=13)
    tokens = torch.tensor([[1, 4, 7, 2, 0, 0, 0, 0, 0]])

    expected = source.forward_mixed_int4_input_attention_q(tokens)
    actual = materialize_mixed_int4_input_attention_q(source)(tokens)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
