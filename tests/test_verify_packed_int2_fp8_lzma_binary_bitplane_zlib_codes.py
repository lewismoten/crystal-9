import torch

from crystal9 import TinyMoEPolicy
from tools.verify_packed_int2_fp8_lzma_binary_bitplane_zlib_codes import build_report


def test_build_report_emits_integrity_bound_binary_lzma_scale_bitplane_zlib_code_artifact(tmp_path):
    source = TinyMoEPolicy(vocab_size=13).eval()
    source_path = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict()}, source_path)

    report = build_report(source_path, tmp_path / "output", ["a"], lambda history: "a", 13)

    assert report["artifact"].endswith("crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-binary-bitplane-zlib-codes.c9i2")
    assert report["acceptance"]["legal_histories"] == 1
    assert report["quantization"]["code_packing"] == "bitplane-transposed"
    assert (tmp_path / "output" / "report.json").is_file()
