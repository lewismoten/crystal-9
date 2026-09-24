from pathlib import Path

import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from tools.verify_packed_int2_fp8_e5m2 import run_preflight


def test_fp8_e5m2_preflight_records_a_packed_scalar_candidate(tmp_path: Path):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source = tmp_path / "f32-reference.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source)

    report = run_preflight(
        source_path=source,
        output_dir=tmp_path / "preflight",
        histories=["", "a", "ab"],
        device=torch.device("cpu"),
    )

    assert report["layout"] == "complete-scalar-group-int2-packed-fp8-e5m2-scales"
    assert report["quantization"]["scale_type"] == "float8_e5m2"
    assert report["quantization"]["scale_count"] == report["parameter_values"]
    assert report["acceptance"]["legal_histories"] == 3
    assert (tmp_path / "preflight" / "report.json").is_file()
