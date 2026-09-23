import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_mixed_int4_attention_bias_qat


def test_attention_bias_qat_changes_only_attention_biases(tmp_path, monkeypatch):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source_path)
    monkeypatch.setattr("training.legal_histories", lambda: ["", "a", "ab"])

    report = run_mixed_int4_attention_bias_qat(
        epochs=1, batch_size=2, source_path=source_path, output_dir=tmp_path / "run", device=torch.device("cpu")
    )

    saved = torch.load(tmp_path / "run" / "artifacts-qat-mixed-int4-row-all-attention-biases.pt", weights_only=False)["state_dict"]
    original = torch.load(source_path, weights_only=False)["state_dict"]
    frozen = [name for name in original if name not in {"attention.in_proj_bias", "attention.out_proj.bias"}]
    assert all(torch.equal(original[name], saved[name]) for name in frozen)
    assert report["layout"] == "mixed-int4-row-all-attention-biases"
    assert report["qat_forward"]["legal_histories"] == 3
    assert report["materialized_mixed_int4_row_all_attention_biases"]["legal_histories"] == 3
