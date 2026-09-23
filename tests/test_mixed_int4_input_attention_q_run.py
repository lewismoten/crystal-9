import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_mixed_int4_input_attention_q_qat


def test_q_only_qat_reports_materialized_runtime_and_preserves_suffix(tmp_path, monkeypatch):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source = TinyMoEPolicy(tokenizer.vocab_size)
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict()}, source_path)
    monkeypatch.setattr("training.legal_histories", lambda: ["", "a", "ab"])

    report = run_mixed_int4_input_attention_q_qat(
        epochs=1,
        batch_size=2,
        source_path=source_path,
        output_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )

    assert report["layout"] == "mixed-int4-row-input-attention-q"
    assert report["qat_forward"]["legal_histories"] == 3
    assert report["materialized_mixed_int4_row_input_attention_q"]["legal_histories"] == 3
