import torch

from crystal9 import quantize_ste


def test_fake_quantization_uses_a_straight_through_gradient():
    weights = torch.tensor([-0.7, 0.2, 0.9], requires_grad=True)

    quantize_ste(weights, 4).sum().backward()

    assert torch.equal(weights.grad, torch.ones_like(weights))
