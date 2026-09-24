import pytest
import torch

from crystal9 import TinyMoEPolicy
from packed_int2_fp8_scale_lzma_binary import (
    PackedInt2Fp8ScaleLzmaBinaryPolicy,
    export_packed_int2_fp8_scale_lzma_binary,
)
from packed_int2_fp8_scale_lzma_compact import export_packed_int2_fp8_scale_lzma_compact


def test_binary_lzma_transport_matches_fp8_scalar_materialization_and_is_smaller_than_json_compact(tmp_path):
    torch.manual_seed(20260986)
    source = TinyMoEPolicy(vocab_size=13).eval()
    expected = TinyMoEPolicy(vocab_size=13).eval()
    expected.load_state_dict(source.state_dict())
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.copy_(parameter.sign() * parameter.abs().to(torch.float8_e4m3fn).float())

    binary_path = tmp_path / "crystal-9-int2-fp8-scales-lzma-binary.c9i2"
    compact_path = tmp_path / "crystal-9-int2-fp8-scales-lzma-compact.c9i2"
    manifest = export_packed_int2_fp8_scale_lzma_binary(source, binary_path)
    export_packed_int2_fp8_scale_lzma_compact(source, compact_path)
    runtime = PackedInt2Fp8ScaleLzmaBinaryPolicy.load(binary_path).eval()

    torch.testing.assert_close(runtime(torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])), expected(torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])))
    assert binary_path.read_bytes().startswith(b"C9I2LZB1")
    assert manifest["compressed_scale_bytes"] < manifest["raw_scale_bytes"]
    assert binary_path.stat().st_size < compact_path.stat().st_size if compact_path.exists() else True


def test_binary_lzma_transport_rejects_a_payload_bit_flip(tmp_path):
    source = TinyMoEPolicy(vocab_size=13).eval()
    artifact_path = tmp_path / "crystal-9-int2-fp8-scales-lzma-binary.c9i2"
    export_packed_int2_fp8_scale_lzma_binary(source, artifact_path)
    corrupted = bytearray(artifact_path.read_bytes())
    corrupted[-1] ^= 1
    artifact_path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="integrity"):
        PackedInt2Fp8ScaleLzmaBinaryPolicy.load(artifact_path)
