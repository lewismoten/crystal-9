import hashlib
from pathlib import Path

from tools.build_int2_research_distribution import build_distribution


def test_build_distribution_binds_artifact_report_and_runtime_with_checksums(tmp_path):
    root = Path(__file__).parents[1]
    artifact = root / "artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes-preflight-20260925/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes.c9i2"
    report = root / "artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes-preflight-20260925/report.json"

    manifest = build_distribution(root, artifact, report, tmp_path / "int2-research-v1")

    assert manifest["classification"] == "staged research artifact; not a deployable INT2 release"
    assert manifest["acceptance"]["policy_misses"] == 0
    assert manifest["acceptance"]["legal_histories"] == 294778
    sums = (tmp_path / "int2-research-v1" / "SHA256SUMS").read_text().splitlines()
    assert sums
    for line in sums:
        digest, filename = line.split("  ")
        assert digest == hashlib.sha256((tmp_path / "int2-research-v1" / filename).read_bytes()).hexdigest()


def test_build_distribution_accepts_an_explicit_runtime_closure(tmp_path):
    root = Path(__file__).parents[1]
    artifact = root / "artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-expert-order-fixed-permutation-stream-binary-bitplane-zlib-codes-preflight-20260928/crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-expert-order-fixed-permutation-stream-binary-bitplane-zlib-codes.c9i2"
    report = root / "artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-expert-order-fixed-permutation-stream-binary-bitplane-zlib-codes-preflight-20260928/report.json"
    runtime_files = (
        "packed_int2_fp8_scale_lzma_expert_order_fixed_permutation_stream_binary_bitplane_zlib_codes.py",
        "packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes.py",
        "packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes.py",
        "packed_int2.py",
        "crystal9.py",
        "design.json",
        "tools/int2_fp8_scale_expert_order_search_hierarchy_screen.py",
    )

    manifest = build_distribution(root, artifact, report, tmp_path / "int2-research-v2", runtime_files=runtime_files)

    assert manifest["artifact"] == artifact.name
    assert manifest["runtime_files"] == [f"runtime/{name}" for name in runtime_files]
    assert (tmp_path / "int2-research-v2" / "runtime" / runtime_files[-1]).is_file()
