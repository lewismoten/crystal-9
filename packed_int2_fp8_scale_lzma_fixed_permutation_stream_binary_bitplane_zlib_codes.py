"""Fixed-permutation stream binary packed INT2 container."""

from __future__ import annotations

import hashlib
import lzma
import zlib
from pathlib import Path

import torch

from crystal9 import TinyMoEPolicy
from packed_int2 import FORMAT as BASE_FORMAT
from packed_int2 import PackedInt2Policy, _encode, _integrity_digest
from packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes import _bitplane_decode, _bitplane_encode
from packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes import _permute_codes

MAGIC_PREFIX = b"C9I2FSP"
FORMAT = "crystal-9-packed-int2-fp8-scale-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes-v1"
DIGEST_BYTES = 32
FIXED_VOCAB_SIZE = 13
FIXED_CODE_PERMUTATION = (0, 3, 1, 2)


def _digest(header: bytes, compressed_scales: bytes, compressed_codes: bytes) -> bytes:
    return hashlib.sha256(header + compressed_scales + compressed_codes).digest()


def _raw_sizes() -> tuple[int, int]:
    prototype = TinyMoEPolicy(FIXED_VOCAB_SIZE)
    values = list(prototype.state_dict().values())
    return sum((value.numel() + 3) // 4 for value in values), sum(value.numel() for value in values)


def export_packed_int2_fp8_scale_lzma_fixed_permutation_stream_binary_bitplane_zlib_codes(source: TinyMoEPolicy, path: str | Path) -> dict:
    if source.embedding.num_embeddings != FIXED_VOCAB_SIZE:
        raise ValueError("fixed-permutation stream binary packed INT2 requires the fixed Crystal-9 vocabulary")
    raw_codes = bytearray()
    raw_scales = bytearray()
    for value in source.state_dict().values():
        record = _encode(value, torch.float8_e4m3fn)
        raw_codes.extend(record["packed"].contiguous().numpy().tobytes())
        raw_scales.extend(record["scales"].contiguous().view(torch.uint8).numpy().tobytes())
    compressed_codes = zlib.compress(_bitplane_encode(_permute_codes(bytes(raw_codes), FIXED_CODE_PERMUTATION)), level=9)
    compressed_scales = lzma.compress(bytes(raw_scales), preset=9)
    digest = _digest(MAGIC_PREFIX, compressed_scales, compressed_codes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MAGIC_PREFIX + digest + compressed_scales + compressed_codes)
    return {
        "format": FORMAT,
        "layout": "complete-scalar-group-int2-fp8-scales-lzma-fixed-permutation-stream-binary-bitplane-zlib-codes-research",
        "parameter_values": len(raw_scales),
        "scale_type": "float8_e4m3fn-lzma",
        "code_permutation": FIXED_CODE_PERMUTATION,
        "raw_code_bytes": len(raw_codes),
        "compressed_code_bytes": len(compressed_codes),
        "raw_scale_bytes": len(raw_scales),
        "compressed_scale_bytes": len(compressed_scales),
        "integrity_sha256": digest.hex(),
    }


def _load(path: str | Path) -> tuple[bytes, bytes]:
    payload = Path(path).read_bytes()
    if len(payload) < len(MAGIC_PREFIX) + DIGEST_BYTES:
        raise ValueError("not a Crystal-9 fixed-permutation stream binary packed INT2 artifact")
    header = payload[:len(MAGIC_PREFIX)]
    if header != MAGIC_PREFIX:
        raise ValueError("invalid fixed-permutation stream binary packed artifact payload")
    digest = payload[len(header):len(header) + DIGEST_BYTES]
    streams = payload[len(header) + DIGEST_BYTES:]
    decoder = lzma.LZMADecompressor()
    raw_scales = decoder.decompress(streams)
    if not decoder.eof or not decoder.unused_data:
        raise ValueError("invalid fixed-permutation stream binary scale payload")
    compressed_codes = decoder.unused_data
    compressed_scales = streams[:len(streams) - len(compressed_codes)]
    if digest != _digest(header, compressed_scales, compressed_codes):
        raise ValueError("packed artifact integrity validation failed")
    raw_code_size, raw_scale_size = _raw_sizes()
    if len(raw_scales) != raw_scale_size:
        raise ValueError("compressed scale payload length validation failed")
    inverse_permutation = (0, 2, 3, 1)
    raw_codes = _permute_codes(_bitplane_decode(zlib.decompress(compressed_codes), raw_code_size), inverse_permutation)
    return raw_codes, raw_scales


class PackedInt2Fp8ScaleLzmaFixedPermutationStreamBinaryBitplaneZlibCodesPolicy:
    def __init__(self, packed: bytes, raw_scales: bytes) -> None:
        prototype = TinyMoEPolicy(FIXED_VOCAB_SIZE)
        tensors = {}
        packed_offset = scale_offset = 0
        for name, value in prototype.state_dict().items():
            packed_count = (value.numel() + 3) // 4
            packed_end = packed_offset + packed_count
            scale_end = scale_offset + value.numel()
            tensors[name] = {
                "shape": tuple(value.shape), "count": value.numel(),
                "packed": torch.frombuffer(bytearray(packed[packed_offset:packed_end]), dtype=torch.uint8).clone(),
                "scales": torch.frombuffer(bytearray(raw_scales[scale_offset:scale_end]), dtype=torch.uint8).view(torch.float8_e4m3fn).clone(),
            }
            packed_offset, scale_offset = packed_end, scale_end
        if packed_offset != len(packed) or scale_offset != len(raw_scales):
            raise ValueError("fixed-permutation stream binary tensor payload validation failed")
        architecture = {"vocab_size": FIXED_VOCAB_SIZE, "hidden_size": prototype.embedding.embedding_dim, "experts": len(prototype.experts), "heads": prototype.attention.num_heads, "norm_eps": prototype.norm.eps}
        manifest = {"format": BASE_FORMAT, "layout": "complete-scalar-group-int2-research", "architecture": architecture, "parameter_values": sum(value.numel() for value in prototype.parameters()), "scale_type": "float8_e4m3fn", "tensors": tensors}
        manifest["integrity_sha256"] = _integrity_digest(manifest)
        self._runtime = PackedInt2Policy(manifest)

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt2Fp8ScaleLzmaFixedPermutationStreamBinaryBitplaneZlibCodesPolicy":
        return cls(*_load(path))

    def eval(self) -> "PackedInt2Fp8ScaleLzmaFixedPermutationStreamBinaryBitplaneZlibCodesPolicy":
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._runtime(token_ids)
