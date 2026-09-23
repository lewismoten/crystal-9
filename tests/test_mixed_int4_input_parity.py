import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_input


def test_int4_input_stage_matches_materialized_model_and_preserves_suffix_weights():
    torch.manual_seed(11)
    source = TinyMoEPolicy(vocab_size=13)
    source.eval()
    token_ids = torch.tensor([[1, 3, 8, 0, 0, 0, 0, 0, 0]])

    before = source.output.weight.detach().clone()
    materialized = materialize_mixed_int4_input(source)
    materialized.eval()

    assert torch.equal(source.output.weight, before)
    assert torch.allclose(
        source.forward_mixed_int4_input(token_ids),
        materialized(token_ids),
        atol=1e-6,
        rtol=0,
    )
