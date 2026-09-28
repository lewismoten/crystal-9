import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "verify_packed_int2_fp8_lzma_expert_interleave_fixed_permutation_stream_binary_bitplane_zlib_codes.py"
    spec = importlib.util.spec_from_file_location("expert_interleave_verify", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expert_interleave_report_builder_writes_an_integrity_bound_artifact(tmp_path):
    module = _module()
    report = module.build_report(Path("artifacts-fp32.pt"), tmp_path / "report", [""], lambda _: "0", 13)

    assert report["quantization"]["scale_layout"] == "corresponding-expert-byte-interleaved"
    assert (tmp_path / "report" / "report.json").is_file()
    assert report["acceptance"]["legal_histories"] == 1
