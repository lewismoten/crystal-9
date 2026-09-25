import pytest
import torch

from crystal9 import TinyMoEPolicy
from packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes import (
    export_packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes,
)
from packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes import (
    PackedInt2Fp8ScaleLzmaBinaryPermutedBitplaneZlibCodesPolicy,
    export_packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes,
)


def test_permuted_bitplane_zlib_code_transport_matches_fp8_scalar_materialization_and_improves_container_size(tmp_path):
    torch.manual_seed(20260988)
    source = TinyMoEPolicy(vocab_size=13).eval()
    expected = TinyMoEPolicy(vocab_size=13).eval()
    expected.load_state_dict(source.state_dict())
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.copy_(parameter.sign() * parameter.abs().to(torch.float8_e4m3fn).float())

    candidate_path = tmp_path / "candidate.c9i2"
    baseline_path = tmp_path / "baseline.c9i2"
    manifest = export_packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes(source, candidate_path)
    export_packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes(source, baseline_path)
    runtime = PackedInt2Fp8ScaleLzmaBinaryPermutedBitplaneZlibCodesPolicy.load(candidate_path).eval()

    inputs = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    torch.testing.assert_close(runtime(inputs), expected(inputs))
    assert candidate_path.read_bytes().startswith(b"C9I2PPZ")
    assert manifest["compressed_code_bytes"] <= 3183
    assert candidate_path.stat().st_size < baseline_path.stat().st_size


def test_permuted_bitplane_zlib_code_transport_rejects_a_payload_bit_flip(tmp_path):
    source = TinyMoEPolicy(vocab_size=13).eval()
    artifact_path = tmp_path / "candidate.c9i2"
    export_packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes(source, artifact_path)
    corrupted = bytearray(artifact_path.read_bytes())
    corrupted[-1] ^= 1
    artifact_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="integrity"):
        PackedInt2Fp8ScaleLzmaBinaryPermutedBitplaneZlibCodesPolicy.load(artifact_path)
