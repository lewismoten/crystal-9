import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_mixed_int4_input_attention_qat


def test_attention_int4_qat_preserves_the_verified_expert_output_suffix(tmp_path):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source = TinyMoEPolicy(tokenizer.vocab_size)
    source_path = tmp_path / "input-int4.pt"
    torch.save({"state_dict": source.state_dict()}, source_path)

    report = run_mixed_int4_input_attention_qat(
        epochs=1,
        batch_size=1,
        source_path=source_path,
        output_dir=tmp_path,
        histories=["a"],
        device=torch.device("cpu"),
    )

    trained = torch.load(tmp_path / "artifacts-qat-mixed-int4-row-input-attention.pt", weights_only=False)["state_dict"]
    assert torch.equal(trained["output.weight"], source.state_dict()["output.weight"])
    assert report["qat_forward"] == report["materialized_mixed_int4_row_input_attention"]
