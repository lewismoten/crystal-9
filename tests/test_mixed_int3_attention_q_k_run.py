import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_mixed_int3_attention_q_k_qat, run_mixed_int3_attention_q_k_group2_qat, run_mixed_int3_attention_q_k_group4_qat


def test_int3_qk_qat_changes_only_k_projection(tmp_path, monkeypatch):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source_path)
    monkeypatch.setattr("training.legal_histories", lambda: ["", "a", "ab"])

    report = run_mixed_int3_attention_q_k_qat(epochs=1, batch_size=2, source_path=source_path, output_dir=tmp_path / "run", device=torch.device("cpu"))

    saved = torch.load(tmp_path / "run" / "artifacts-qat-mixed-int3-scalar-input-attention-q-k.pt", weights_only=False)["state_dict"]
    original = torch.load(source_path, weights_only=False)["state_dict"]
    width = original["attention.in_proj_weight"].shape[1]
    assert torch.equal(original["attention.in_proj_weight"][:width], saved["attention.in_proj_weight"][:width])
    assert torch.equal(original["attention.in_proj_weight"][2 * width:], saved["attention.in_proj_weight"][2 * width:])
    assert all(torch.equal(original[name], saved[name]) for name in original if name != "attention.in_proj_weight")
    assert report["trainable_tensors"] == ["attention.in_proj_weight[K]"]
    assert report["acceptance"]["legal_histories"] == 3


def test_int3_qk_group4_qat_changes_only_k_projection(tmp_path, monkeypatch):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source_path)
    monkeypatch.setattr("training.legal_histories", lambda: ["", "a", "ab"])

    report = run_mixed_int3_attention_q_k_group4_qat(epochs=1, batch_size=2, source_path=source_path, output_dir=tmp_path / "run", device=torch.device("cpu"))

    saved = torch.load(tmp_path / "run" / "artifacts-qat-mixed-int3-scalar-input-attention-q-k-group4.pt", weights_only=False)["state_dict"]
    original = torch.load(source_path, weights_only=False)["state_dict"]
    width = original["attention.in_proj_weight"].shape[1]
    assert torch.equal(original["attention.in_proj_weight"][:width], saved["attention.in_proj_weight"][:width])
    assert torch.equal(original["attention.in_proj_weight"][2 * width:], saved["attention.in_proj_weight"][2 * width:])
    assert all(torch.equal(original[name], saved[name]) for name in original if name != "attention.in_proj_weight")
    assert report["trainable_tensors"] == ["attention.in_proj_weight[K]"]
    assert report["acceptance"]["legal_histories"] == 3


def test_int3_qk_group2_qat_changes_only_k_projection(tmp_path, monkeypatch):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source_path)
    monkeypatch.setattr("training.legal_histories", lambda: ["", "a", "ab"])

    report = run_mixed_int3_attention_q_k_group2_qat(epochs=1, batch_size=2, source_path=source_path, output_dir=tmp_path / "run", device=torch.device("cpu"))

    saved = torch.load(tmp_path / "run" / "artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2.pt", weights_only=False)["state_dict"]
    original = torch.load(source_path, weights_only=False)["state_dict"]
    width = original["attention.in_proj_weight"].shape[1]
    assert torch.equal(original["attention.in_proj_weight"][:width], saved["attention.in_proj_weight"][:width])
    assert torch.equal(original["attention.in_proj_weight"][2 * width:], saved["attention.in_proj_weight"][2 * width:])
    assert all(torch.equal(original[name], saved[name]) for name in original if name != "attention.in_proj_weight")
    assert report["trainable_tensors"] == ["attention.in_proj_weight[K]"]
    assert report["quantization"]["K"]["group_size"] == 2
