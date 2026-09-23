import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_input_attention_v


def test_v_projection_int4_forward_matches_materialized_runtime():
    torch.manual_seed(19)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    tokens = torch.tensor([[1, 4, 7, 2, 0, 0, 0, 0, 0]])

    expected = source.forward_mixed_int4_input_attention_groups(tokens, frozenset({"q", "v"}))
    actual = materialize_mixed_int4_input_attention_v(source)(tokens)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
