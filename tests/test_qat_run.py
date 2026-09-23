import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from training import run_qat


def test_qat_run_writes_a_checkpoint_and_report_for_a_bounded_corpus(tmp_path):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source = TinyMoEPolicy(tokenizer.vocab_size)
    source_path = tmp_path / "fp32.pt"
    torch.save({"state_dict": source.state_dict()}, source_path)

    result = run_qat(
        bits=4,
        epochs=1,
        batch_size=1,
        source_path=source_path,
        output_dir=tmp_path,
        histories=["a"],
        device=torch.device("cpu"),
    )

    assert result["bits"] == 4
    assert (tmp_path / "artifacts-qat-int4.pt").exists()
    assert (tmp_path / "qat-int4-report.json").exists()
