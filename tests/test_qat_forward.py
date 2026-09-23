import torch

from crystal9 import TinyMoEPolicy


def test_quantized_forward_keeps_master_weights_trainable():
    model = TinyMoEPolicy(vocab_size=13)
    tokens = torch.tensor([[1, 4, 0, 0]])

    model.forward_quantized(tokens, bits=4).sum().backward()

    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()
