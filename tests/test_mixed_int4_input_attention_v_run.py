import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_mixed_int4_input_attention_v_qat


def test_q_v_qat_reports_materialized_runtime(tmp_path, monkeypatch):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source_path)
    monkeypatch.setattr("training.legal_histories", lambda: ["", "a", "ab"])

    report = run_mixed_int4_input_attention_v_qat(
        epochs=1, batch_size=2, source_path=source_path, output_dir=tmp_path / "run", device=torch.device("cpu")
    )

    assert report["layout"] == "mixed-int4-row-input-attention-q-v"
    assert report["qat_forward"]["legal_histories"] == 3
    assert report["materialized_mixed_int4_row_input_attention_q_v"]["legal_histories"] == 3
