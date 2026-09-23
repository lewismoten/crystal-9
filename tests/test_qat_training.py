import torch

from crystal9 import TinyMoEPolicy
from training import qat_step


def test_qat_step_updates_master_weights_through_int4_forward_path():
    torch.manual_seed(2)
    model = TinyMoEPolicy(vocab_size=13)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.output.weight.detach().clone()

    loss = qat_step(model, optimizer, torch.tensor([[1, 4, 0, 0]]), torch.tensor([5]), bits=4)

    assert loss > 0
    assert not torch.equal(before, model.output.weight.detach())
