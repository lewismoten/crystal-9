import torch

import crystal9
from crystal9 import TinyMoEPolicy


def test_int2_scalar_suffix_through_expert_biases_and_position_fake_qat_matches_materialized_model():
    torch.manual_seed(20260966)
    source = TinyMoEPolicy(vocab_size=13).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])

    expected = source.forward_mixed_int2_scalar_suffix_output_bias_router_weight_bias_expert_biases_position(token_ids)
    actual = crystal9.materialize_mixed_int2_scalar_suffix_output_bias_router_weight_bias_expert_biases_position(source)(token_ids)

    torch.testing.assert_close(actual, expected)
