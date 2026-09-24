import pytest
import torch

from crystal9 import TinyMoEPolicy
from packed_int2_fp8_scale_lzma_binary import export_packed_int2_fp8_scale_lzma_binary
from packed_int2_fp8_scale_lzma_binary_zlib_codes import (
    PackedInt2Fp8ScaleLzmaBinaryZlibCodesPolicy,
    export_packed_int2_fp8_scale_lzma_binary_zlib_codes,
)


def test_binary_zlib_code_transport_matches_fp8_scalar_materialization_and_is_smaller_than_binary(tmp_path):
    torch.manual_seed(20260987)
    source = TinyMoEPolicy(vocab_size=13).eval()
    expected = TinyMoEPolicy(vocab_size=13).eval()
    expected.load_state_dict(source.state_dict())
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.copy_(parameter.sign() * parameter.abs().to(torch.float8_e4m3fn).float())

    candidate_path = tmp_path / "crystal-9-int2-fp8-scales-lzma-binary-zlib-codes.c9i2"
    binary_path = tmp_path / "crystal-9-int2-fp8-scales-lzma-binary.c9i2"
    manifest = export_packed_int2_fp8_scale_lzma_binary_zlib_codes(source, candidate_path)
    export_packed_int2_fp8_scale_lzma_binary(source, binary_path)
    runtime = PackedInt2Fp8ScaleLzmaBinaryZlibCodesPolicy.load(candidate_path).eval()

    inputs = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    torch.testing.assert_close(runtime(inputs), expected(inputs))
    assert candidate_path.read_bytes().startswith(b"C9I2LZC1")
    assert manifest["compressed_code_bytes"] < manifest["raw_code_bytes"]
    assert candidate_path.stat().st_size < binary_path.stat().st_size


def test_binary_zlib_code_transport_rejects_a_payload_bit_flip(tmp_path):
    source = TinyMoEPolicy(vocab_size=13).eval()
    artifact_path = tmp_path / "crystal-9-int2-fp8-scales-lzma-binary-zlib-codes.c9i2"
    export_packed_int2_fp8_scale_lzma_binary_zlib_codes(source, artifact_path)
    corrupted = bytearray(artifact_path.read_bytes())
    corrupted[-1] ^= 1
    artifact_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="integrity"):
        PackedInt2Fp8ScaleLzmaBinaryZlibCodesPolicy.load(artifact_path)
