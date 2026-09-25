from pathlib import Path

from tools.verify_packed_int2_fp8_lzma_fixed_permutation_stream_binary_bitplane_zlib_codes import build_report


def test_fixed_permutation_stream_preflight_report_records_exact_gate(tmp_path):
    source = Path(__file__).parents[1] / "artifacts-fp32.pt"
    report = build_report(source, tmp_path / "report", [""], lambda _history: "a", 13)

    assert report["decision"] == "rejected"
    assert report["quantization"]["code_permutation"] == [0, 3, 1, 2]
    assert report["artifact_bytes"] > 0
