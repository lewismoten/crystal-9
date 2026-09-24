import torch

import crystal9
from crystal9 import TinyMoEPolicy


def test_int3_accepted_scope_and_group2_position_fake_qat_match_materialized_model():
    torch.manual_seed(104)
    source = TinyMoEPolicy(vocab_size=13).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])

    expected = source.forward_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group2(token_ids)
    actual = crystal9.materialize_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group2(source)(token_ids)

    torch.testing.assert_close(actual, expected)
