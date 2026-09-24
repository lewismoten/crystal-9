import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_mixed_int3_attention_q_qat


def test_int3_q_qat_changes_only_q_projection(tmp_path, monkeypatch):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source_path)
    monkeypatch.setattr("training.legal_histories", lambda: ["", "a", "ab"])

    report = run_mixed_int3_attention_q_qat(epochs=1, batch_size=2, source_path=source_path, output_dir=tmp_path / "run", device=torch.device("cpu"))

    saved = torch.load(tmp_path / "run" / "artifacts-qat-mixed-int3-scalar-input-attention-q.pt", weights_only=False)["state_dict"]
    original = torch.load(source_path, weights_only=False)["state_dict"]
    width = original["attention.in_proj_weight"].shape[1]
    assert torch.equal(original["attention.in_proj_weight"][width:], saved["attention.in_proj_weight"][width:])
    assert all(torch.equal(original[name], saved[name]) for name in original if name != "attention.in_proj_weight")
    assert report["trainable_tensors"] == ["attention.in_proj_weight[Q]"]
    assert report["acceptance"]["legal_histories"] == 3
