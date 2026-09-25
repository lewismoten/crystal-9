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
