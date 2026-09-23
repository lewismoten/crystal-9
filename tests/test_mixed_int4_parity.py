import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4


def test_mixed_int4_qat_forward_matches_the_materialized_mixed_model():
    torch.manual_seed(7)
    source = TinyMoEPolicy(vocab_size=13)
    tokens = torch.tensor([[1, 4, 8, 0], [1, 3, 0, 0]])

    expected = source.forward_mixed_int4(tokens)
    materialized = materialize_mixed_int4(source)

    assert torch.allclose(expected, materialized(tokens), atol=1e-6, rtol=0)
