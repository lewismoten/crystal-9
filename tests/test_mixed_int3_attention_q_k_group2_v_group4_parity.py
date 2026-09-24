import torch

import crystal9
from crystal9 import GameTokenizer


def test_int3_q_rowwise_k_group2_v_group4_fake_qat_matches_materialized_runtime():
    tokenizer = GameTokenizer.from_design_file("design.json")
    source = crystal9.TinyMoEPolicy(tokenizer.vocab_size).eval()
    token_ids = torch.tensor([[1, 4, 7, 0], [1, 3, 8, 11]])

    expected = source.forward_mixed_int3_scalar_input_attention_q_k_group2_v_group4(token_ids)
    actual = crystal9.materialize_mixed_int3_scalar_input_attention_q_k_group2_v_group4(source)(token_ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
