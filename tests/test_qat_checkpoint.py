import torch

from crystal9 import TinyMoEPolicy
from training import load_reference_model


def test_reference_loader_restores_a_saved_fp32_state_dict(tmp_path):
    source = TinyMoEPolicy(vocab_size=13)
    path = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict()}, path)

    restored = load_reference_model(path, vocab_size=13, device=torch.device("cpu"))

    assert torch.equal(restored.output.weight, source.output.weight)
