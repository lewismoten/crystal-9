"""Fixed-architecture binary packed INT2 container with permuted bitplane-zlib codes."""

from __future__ import annotations

import hashlib
import lzma
import struct
import zlib
from pathlib import Path

import torch

from crystal9 import TinyMoEPolicy
from packed_int2 import FORMAT as BASE_FORMAT
from packed_int2 import PackedInt2Policy, _encode, _integrity_digest
from packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes import _bitplane_decode, _bitplane_encode
from packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes import CODE_PERMUTATIONS, _permute_codes

MAGIC_PREFIX = b"C9I2FPZ"
FORMAT = "crystal-9-packed-int2-fp8-scale-lzma-fixed-binary-permuted-bitplane-zlib-codes-v1"
HEADER = struct.Struct(">8sIIIII")
DIGEST_BYTES = 32


def _digest(header: bytes, compressed_codes: bytes, compressed_scales: bytes) -> bytes:
    return hashlib.sha256(header + compressed_codes + compressed_scales).digest()


def export_packed_int2_fp8_scale_lzma_fixed_binary_permuted_bitplane_zlib_codes(source: TinyMoEPolicy, path: str | Path) -> dict:
    raw_codes = bytearray()
    raw_scales = bytearray()
    for value in source.state_dict().values():
        record = _encode(value, torch.float8_e4m3fn)
        raw_codes.extend(record["packed"].contiguous().numpy().tobytes())
        raw_scales.extend(record["scales"].contiguous().view(torch.uint8).numpy().tobytes())
    candidates = [
        (zlib.compress(_bitplane_encode(_permute_codes(bytes(raw_codes), permutation)), level=9), index)
        for index, permutation in enumerate(CODE_PERMUTATIONS)
    ]
    compressed_codes, permutation_index = min(candidates, key=lambda candidate: len(candidate[0]))
    compressed_scales = lzma.compress(bytes(raw_scales), preset=9)
    header = HEADER.pack(
        MAGIC_PREFIX + bytes((permutation_index,)), source.embedding.num_embeddings,
        len(raw_codes), len(compressed_codes), len(raw_scales), len(compressed_scales),
    )
    digest = _digest(header, compressed_codes, compressed_scales)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + digest + compressed_codes + compressed_scales)
    return {
        "format": FORMAT,
        "layout": "complete-scalar-group-int2-fp8-scales-lzma-fixed-binary-permuted-bitplane-zlib-codes-research",
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "scale_type": "float8_e4m3fn-lzma",
        "code_permutation": CODE_PERMUTATIONS[permutation_index],
        "raw_code_bytes": len(raw_codes),
        "compressed_code_bytes": len(compressed_codes),
        "raw_scale_bytes": len(raw_scales),
        "compressed_scale_bytes": len(compressed_scales),
        "integrity_sha256": digest.hex(),
    }


def _load(path: str | Path) -> tuple[int, bytes, bytes, int]:
    payload = Path(path).read_bytes()
    if len(payload) < HEADER.size + DIGEST_BYTES:
        raise ValueError("not a Crystal-9 fixed binary packed INT2 artifact")
    magic, vocab_size, raw_code_size, compressed_code_size, raw_scale_size, compressed_scale_size = HEADER.unpack(payload[:HEADER.size])
    expected_size = HEADER.size + DIGEST_BYTES + compressed_code_size + compressed_scale_size
    if not magic.startswith(MAGIC_PREFIX) or magic[-1] >= len(CODE_PERMUTATIONS) or len(payload) != expected_size:
        raise ValueError("invalid fixed binary packed artifact payload")
    digest_start = HEADER.size
    codes_start = digest_start + DIGEST_BYTES
    codes_end = codes_start + compressed_code_size
    digest = payload[digest_start:codes_start]
    header = payload[:HEADER.size]
    compressed_codes = payload[codes_start:codes_end]
    compressed_scales = payload[codes_end:]
    if digest != _digest(header, compressed_codes, compressed_scales):
        raise ValueError("packed artifact integrity validation failed")
    permutation = CODE_PERMUTATIONS[magic[-1]]
    inverse_permutation = tuple(permutation.index(code) for code in range(4))
    raw_codes = _permute_codes(_bitplane_decode(zlib.decompress(compressed_codes), raw_code_size), inverse_permutation)
    return vocab_size, raw_codes, compressed_scales, raw_scale_size


class PackedInt2Fp8ScaleLzmaFixedBinaryPermutedBitplaneZlibCodesPolicy:
    def __init__(self, vocab_size: int, packed: bytes, compressed_scales: bytes, raw_scale_size: int) -> None:
        raw_scales = lzma.decompress(compressed_scales)
        if len(raw_scales) != raw_scale_size:
            raise ValueError("compressed scale payload length validation failed")
        prototype = TinyMoEPolicy(vocab_size)
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
            raise ValueError("fixed binary tensor payload validation failed")
        architecture = {"vocab_size": vocab_size, "hidden_size": prototype.embedding.embedding_dim, "experts": len(prototype.experts), "heads": prototype.attention.num_heads, "norm_eps": prototype.norm.eps}
        manifest = {"format": BASE_FORMAT, "layout": "complete-scalar-group-int2-research", "architecture": architecture, "parameter_values": sum(value.numel() for value in prototype.parameters()), "scale_type": "float8_e4m3fn", "tensors": tensors}
        manifest["integrity_sha256"] = _integrity_digest(manifest)
        self._runtime = PackedInt2Policy(manifest)

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt2Fp8ScaleLzmaFixedBinaryPermutedBitplaneZlibCodesPolicy":
        return cls(*_load(path))

    def eval(self) -> "PackedInt2Fp8ScaleLzmaFixedBinaryPermutedBitplaneZlibCodesPolicy":
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._runtime(token_ids)
