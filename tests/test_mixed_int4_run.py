import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_mixed_int4_qat


def test_mixed_int4_qat_reports_identical_qat_and_materialized_policy_counts(tmp_path):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source_path = tmp_path / "fp32.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source_path)

    report = run_mixed_int4_qat(
        epochs=1,
        batch_size=1,
        source_path=source_path,
        output_dir=tmp_path,
        histories=["a"],
        device=torch.device("cpu"),
    )

    assert report["qat_forward"] == report["materialized_mixed_int4"]
