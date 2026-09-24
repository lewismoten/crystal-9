from pathlib import Path

import torch

from tools.verify_packed_int2_group2_fp16 import run_preflight


ROOT = Path(__file__).resolve().parents[1]


def test_group2_fp16_preflight_records_independent_runtime_result(tmp_path):
    report = run_preflight(
        ROOT / "artifacts-fp32.pt",
        tmp_path / "candidate",
        [""],
        torch.device("cpu"),
    )

    assert report["quantization"]["group_size"] == 2
    assert report["quantization"]["scale_type"] == "float16"
    assert report["acceptance"]["legal_histories"] == 1
    assert (tmp_path / "candidate" / "report.json").is_file()
