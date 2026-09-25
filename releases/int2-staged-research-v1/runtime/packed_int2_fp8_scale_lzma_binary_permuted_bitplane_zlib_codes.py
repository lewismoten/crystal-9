"""Canonical packed INT2 container with permuted bitplane-zlib codes and LZMA scales."""

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

MAGIC_PREFIX = b"C9I2PPZ"
FORMAT = "crystal-9-packed-int2-fp8-scale-lzma-binary-permuted-bitplane-zlib-codes-v1"
HEADER = struct.Struct(">8sIIIIfIIII")
DIGEST_BYTES = 32
CODE_PERMUTATIONS = (
    (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1),
    (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2),
    (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
    (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0),
    (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1),
    (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0),
)


def _digest(header: bytes, compressed_codes: bytes, compressed_scales: bytes) -> bytes:
    return hashlib.sha256(header + compressed_codes + compressed_scales).digest()


def _permute_codes(raw_codes: bytes, permutation: tuple[int, int, int, int]) -> bytes:
    return bytes(
        sum(permutation[(packed >> shift) & 3] << shift for shift in range(0, 8, 2))
        for packed in raw_codes
    )


def _architecture(source: TinyMoEPolicy) -> dict:
    return {
        "vocab_size": source.embedding.num_embeddings,
        "hidden_size": source.embedding.embedding_dim,
        "experts": len(source.experts),
        "heads": source.attention.num_heads,
        "norm_eps": source.norm.eps,
    }


def export_packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes(source: TinyMoEPolicy, path: str | Path) -> dict:
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
    architecture = _architecture(source)
    header = HEADER.pack(
        MAGIC_PREFIX + bytes((permutation_index,)), architecture["vocab_size"], architecture["hidden_size"], architecture["experts"],
        architecture["heads"], architecture["norm_eps"], len(raw_codes), len(compressed_codes),
        len(raw_scales), len(compressed_scales),
    )
    digest = _digest(header, compressed_codes, compressed_scales)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + digest + compressed_codes + compressed_scales)
    return {
        "format": FORMAT,
        "layout": "complete-scalar-group-int2-fp8-scales-lzma-binary-permuted-bitplane-zlib-codes-research",
        "architecture": architecture,
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "scale_type": "float8_e4m3fn-lzma",
        "code_permutation": CODE_PERMUTATIONS[permutation_index],
        "raw_code_bytes": len(raw_codes),
        "compressed_code_bytes": len(compressed_codes),
        "raw_scale_bytes": len(raw_scales),
        "compressed_scale_bytes": len(compressed_scales),
        "integrity_sha256": digest.hex(),
    }


def _load(path: str | Path) -> tuple[dict, bytes, bytes]:
    payload = Path(path).read_bytes()
    if len(payload) < HEADER.size + DIGEST_BYTES:
        raise ValueError("not a Crystal-9 binary permuted-bitplane-code packed INT2 artifact")
    magic, vocab_size, hidden_size, experts, heads, norm_eps, raw_code_size, compressed_code_size, raw_scale_size, compressed_scale_size = HEADER.unpack(payload[:HEADER.size])
    expected_size = HEADER.size + DIGEST_BYTES + compressed_code_size + compressed_scale_size
    if not magic.startswith(MAGIC_PREFIX) or magic[-1] >= len(CODE_PERMUTATIONS) or len(payload) != expected_size:
        raise ValueError("invalid binary packed artifact payload")
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
    permuted_codes = _bitplane_decode(zlib.decompress(compressed_codes), raw_code_size)
    raw_codes = _permute_codes(permuted_codes, inverse_permutation)
    return {"architecture": {"vocab_size": vocab_size, "hidden_size": hidden_size, "experts": experts, "heads": heads, "norm_eps": norm_eps}, "raw_scale_bytes": raw_scale_size}, raw_codes, compressed_scales


class PackedInt2Fp8ScaleLzmaBinaryPermutedBitplaneZlibCodesPolicy:
    def __init__(self, header: dict, packed: bytes, compressed_scales: bytes) -> None:
        raw_scales = lzma.decompress(compressed_scales)
        if len(raw_scales) != header["raw_scale_bytes"]:
            raise ValueError("compressed scale payload length validation failed")
        architecture = header["architecture"]
        prototype = TinyMoEPolicy(architecture["vocab_size"])
        if (prototype.embedding.embedding_dim != architecture["hidden_size"] or len(prototype.experts) != architecture["experts"] or prototype.attention.num_heads != architecture["heads"] or abs(prototype.norm.eps - architecture["norm_eps"]) > 1e-10):
            raise ValueError("unsupported canonical binary architecture")
        tensors = {}
        packed_offset = scale_offset = 0
        for name, value in prototype.state_dict().items():
            packed_count = (value.numel() + 3) // 4
            scale_count = value.numel()
            packed_end = packed_offset + packed_count
            scale_end = scale_offset + scale_count
            tensors[name] = {
                "shape": tuple(value.shape), "count": value.numel(),
                "packed": torch.frombuffer(bytearray(packed[packed_offset:packed_end]), dtype=torch.uint8).clone(),
                "scales": torch.frombuffer(bytearray(raw_scales[scale_offset:scale_end]), dtype=torch.uint8).view(torch.float8_e4m3fn).clone(),
            }
            packed_offset, scale_offset = packed_end, scale_end
        if packed_offset != len(packed) or scale_offset != len(raw_scales):
            raise ValueError("canonical binary tensor payload validation failed")
        manifest = {"format": BASE_FORMAT, "layout": "complete-scalar-group-int2-research", "architecture": architecture, "parameter_values": sum(value.numel() for value in prototype.parameters()), "scale_type": "float8_e4m3fn", "tensors": tensors}
        manifest["integrity_sha256"] = _integrity_digest(manifest)
        self._runtime = PackedInt2Policy(manifest)

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt2Fp8ScaleLzmaBinaryPermutedBitplaneZlibCodesPolicy":
        return cls(*_load(path))

    def eval(self) -> "PackedInt2Fp8ScaleLzmaBinaryPermutedBitplaneZlibCodesPolicy":
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._runtime(token_ids)
