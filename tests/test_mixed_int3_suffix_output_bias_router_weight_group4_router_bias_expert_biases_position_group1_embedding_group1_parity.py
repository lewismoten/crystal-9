from __future__ import annotations

import torch

import crystal9
from crystal9 import GameTokenizer


def test_int3_scalar_group_embedding_fake_qat_matches_materialized_runtime():
    tokenizer = GameTokenizer.from_design_file("design.json")
    source = crystal9.TinyMoEPolicy(tokenizer.vocab_size).eval()
    token_ids = torch.tensor([[1, 4, 7, 0], [1, 3, 8, 11]])

    expected = source.forward_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group1_embedding_group1(token_ids)
    actual = crystal9.materialize_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_position_group1_embedding_group1(source)(token_ids)

    assert torch.equal(expected, actual)
