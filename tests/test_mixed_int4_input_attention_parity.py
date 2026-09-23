import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_input_attention


def test_int4_attention_stage_matches_its_materialized_runtime():
    torch.manual_seed(13)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    token_ids = torch.tensor([[1, 3, 8, 5, 0, 0, 0, 0, 0]])

    materialized = materialize_mixed_int4_input_attention(source)
    materialized.eval()

    assert torch.allclose(
        source.forward_mixed_int4_input_attention(token_ids),
        materialized(token_ids),
        atol=1e-6,
        rtol=0,
    )
