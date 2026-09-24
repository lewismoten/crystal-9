import torch

import crystal9
from crystal9 import GameTokenizer


def test_int3_q_projection_fake_qat_matches_materialized_runtime():
    torch.manual_seed(20260949)
    tokenizer = GameTokenizer.from_design_file("design.json")
    source = crystal9.TinyMoEPolicy(tokenizer.vocab_size).eval()
    token_ids = torch.tensor([[1, 4, 7, 0], [1, 3, 8, 11]])

    expected = source.forward_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group1_embedding_group1_attention_q(token_ids)
    actual = crystal9.materialize_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group1_embedding_group1_attention_q(source)(token_ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
