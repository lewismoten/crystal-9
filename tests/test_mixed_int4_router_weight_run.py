import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_mixed_int4_router_weight_qat


def test_router_weight_qat_changes_only_router_weight(tmp_path, monkeypatch):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source_path)
    monkeypatch.setattr("training.legal_histories", lambda: ["", "a", "ab"])

    report = run_mixed_int4_router_weight_qat(
        epochs=1, batch_size=2, source_path=source_path, output_dir=tmp_path / "run", device=torch.device("cpu")
    )

    saved = torch.load(tmp_path / "run" / "artifacts-qat-mixed-int4-row-input-attention-q-v-out-k-output-bias-router-weight.pt", weights_only=False)["state_dict"]
    original = torch.load(source_path, weights_only=False)["state_dict"]
    frozen = [name for name in original if name != "router.weight"]
    assert all(torch.equal(original[name], saved[name]) for name in frozen)
    assert report["layout"] == "mixed-int4-row-input-attention-q-v-out-k-output-bias-router-weight"
    assert report["qat_forward"]["legal_histories"] == 3
    assert report["materialized_mixed_int4_row_input_attention_q_v_out_k_output_bias_router_weight"]["legal_histories"] == 3
