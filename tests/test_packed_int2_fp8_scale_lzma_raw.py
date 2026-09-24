import pytest
import torch

from crystal9 import TinyMoEPolicy
from packed_int2_fp8_scale_lzma_raw import (
    PackedInt2Fp8ScaleLzmaRawPolicy,
    export_packed_int2_fp8_scale_lzma_raw,
)


def test_raw_lzma_encoded_fp8_scales_runtime_matches_fp8_scalar_materialization(tmp_path):
    torch.manual_seed(20260985)
    source = TinyMoEPolicy(vocab_size=13).eval()
    expected = TinyMoEPolicy(vocab_size=13).eval()
    expected.load_state_dict(source.state_dict())
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.copy_(parameter.sign() * parameter.abs().to(torch.float8_e4m3fn).float())

    artifact_path = tmp_path / "crystal-9-int2-fp8-scales-lzma-raw.pt"
    manifest = export_packed_int2_fp8_scale_lzma_raw(source, artifact_path)
    runtime = PackedInt2Fp8ScaleLzmaRawPolicy.load(artifact_path).eval()

    tokens = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    torch.testing.assert_close(runtime(tokens), expected(tokens))
    assert manifest["scale_type"] == "float8_e4m3fn-lzma-raw"
    assert manifest["compressed_scale_bytes"] < manifest["raw_scale_bytes"]


def test_raw_lzma_encoded_fp8_scales_reject_a_payload_bit_flip(tmp_path):
    source = TinyMoEPolicy(vocab_size=13).eval()
    artifact_path = tmp_path / "crystal-9-int2-fp8-scales-lzma-raw.pt"
    export_packed_int2_fp8_scale_lzma_raw(source, artifact_path)
    manifest = torch.load(artifact_path, map_location="cpu", weights_only=True)
    corrupted = bytearray(manifest["compressed_scales"])
    corrupted[0] ^= 1
    manifest["compressed_scales"] = bytes(corrupted)
    torch.save(manifest, artifact_path)

    with pytest.raises(ValueError, match="integrity"):
        PackedInt2Fp8ScaleLzmaRawPolicy.load(artifact_path)
