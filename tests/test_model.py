import torch

from crystal9 import TinyMoEPolicy, quantize_tensor


def test_tiny_moe_produces_a_logit_for_every_custom_token():
    model = TinyMoEPolicy(vocab_size=13)
    tokens = torch.tensor([[1, 4, 8, 0, 0, 0, 0, 0, 0]])
    logits = model(tokens)

    assert logits.shape == (1, 13)


def test_simulated_quantization_preserves_fp32_and_has_bounded_one_bit_levels():
    values = torch.tensor([-0.8, -0.2, 0.2, 0.8])

    assert torch.equal(quantize_tensor(values, 32), values)
    assert torch.unique(quantize_tensor(values, 1)).numel() <= 2
