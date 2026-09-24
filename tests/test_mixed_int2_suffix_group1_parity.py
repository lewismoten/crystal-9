import torch

import crystal9
from crystal9 import TinyMoEPolicy


def test_int2_suffix_group1_fake_qat_matches_materialized_model():
    torch.manual_seed(20260961)
    source = TinyMoEPolicy(vocab_size=13).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])

    expected = source.forward_mixed_int2_suffix_group1(token_ids)
    actual = crystal9.materialize_mixed_int2_suffix_group1(source)(token_ids)

    torch.testing.assert_close(actual, expected)
