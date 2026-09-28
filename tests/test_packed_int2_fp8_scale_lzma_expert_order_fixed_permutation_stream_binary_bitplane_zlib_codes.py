from pathlib import Path

import pytest
import torch

from crystal9 import TinyMoEPolicy


def _module():
    import importlib.util

    path = Path(__file__).parents[1] / "packed_int2_fp8_scale_lzma_expert_order_fixed_permutation_stream_binary_bitplane_zlib_codes.py"
    spec = importlib.util.spec_from_file_location("expert_order_runtime", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expert_order_runtime_materializes_exact_scalar_fp8_values(tmp_path):
    module = _module()
    checkpoint = torch.load(Path(__file__).parents[1] / "artifacts-fp32.pt", map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size=13).eval()
    source.load_state_dict(checkpoint["state_dict"])
    expected = TinyMoEPolicy(vocab_size=13).eval()
    expected.load_state_dict(source.state_dict())
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.copy_(parameter.sign() * parameter.abs().to(torch.float8_e4m3fn).float())

    artifact = tmp_path / "candidate.c9i2"
    module.export_packed_int2_fp8_scale_lzma_expert_order_fixed_permutation_stream_binary_bitplane_zlib_codes(source, artifact)
    runtime = module.PackedInt2Fp8ScaleLzmaExpertOrderFixedPermutationStreamBinaryBitplaneZlibCodesPolicy.load(artifact).eval()

    tokens = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    torch.testing.assert_close(runtime(tokens), expected(tokens))


def test_expert_order_runtime_rejects_payload_bit_flip(tmp_path):
    module = _module()
    artifact = tmp_path / "candidate.c9i2"
    module.export_packed_int2_fp8_scale_lzma_expert_order_fixed_permutation_stream_binary_bitplane_zlib_codes(TinyMoEPolicy(vocab_size=13).eval(), artifact)
    corrupted = bytearray(artifact.read_bytes())
    corrupted[-1] ^= 1
    artifact.write_bytes(corrupted)

    with pytest.raises(ValueError, match="integrity"):
        module.PackedInt2Fp8ScaleLzmaExpertOrderFixedPermutationStreamBinaryBitplaneZlibCodesPolicy.load(artifact)
