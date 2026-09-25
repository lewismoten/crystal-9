"""Canonical packed INT2 container with bitplane-zlib codes and LZMA scales."""

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

MAGIC = b"C9I2BPZ1"
FORMAT = "crystal-9-packed-int2-fp8-scale-lzma-binary-bitplane-zlib-codes-v1"
HEADER = struct.Struct(">8sIIIIfIIII")
DIGEST_BYTES = 32


def _digest(header: bytes, compressed_codes: bytes, compressed_scales: bytes) -> bytes:
    return hashlib.sha256(header + compressed_codes + compressed_scales).digest()


def _bitplane_encode(raw_codes: bytes) -> bytes:
    code_count = len(raw_codes) * 4
    plane_bytes = (code_count + 7) // 8
    low = bytearray(plane_bytes)
    high = bytearray(plane_bytes)
    for byte_index, packed in enumerate(raw_codes):
        for shift in range(0, 8, 2):
            code_index = byte_index * 4 + shift // 2
            code = (packed >> shift) & 0b11
            if code & 1:
                low[code_index // 8] |= 1 << (code_index % 8)
            if code & 2:
                high[code_index // 8] |= 1 << (code_index % 8)
    return bytes(low + high)


def _bitplane_decode(bitplanes: bytes, raw_code_size: int) -> bytes:
    code_count = raw_code_size * 4
    plane_bytes = (code_count + 7) // 8
    if len(bitplanes) != plane_bytes * 2:
        raise ValueError("bitplane code payload length validation failed")
    low, high = bitplanes[:plane_bytes], bitplanes[plane_bytes:]
    raw = bytearray(raw_code_size)
    for code_index in range(code_count):
        code = ((low[code_index // 8] >> (code_index % 8)) & 1) | (((high[code_index // 8] >> (code_index % 8)) & 1) << 1)
        raw[code_index // 4] |= code << ((code_index % 4) * 2)
    return bytes(raw)


def _architecture(source: TinyMoEPolicy) -> dict:
    return {
        "vocab_size": source.embedding.num_embeddings,
        "hidden_size": source.embedding.embedding_dim,
        "experts": len(source.experts),
        "heads": source.attention.num_heads,
        "norm_eps": source.norm.eps,
    }


def export_packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes(source: TinyMoEPolicy, path: str | Path) -> dict:
    raw_codes = bytearray()
    raw_scales = bytearray()
    for value in source.state_dict().values():
        record = _encode(value, torch.float8_e4m3fn)
        raw_codes.extend(record["packed"].contiguous().numpy().tobytes())
        raw_scales.extend(record["scales"].contiguous().view(torch.uint8).numpy().tobytes())
    bitplane_codes = _bitplane_encode(bytes(raw_codes))
    compressed_codes = zlib.compress(bitplane_codes, level=9)
    compressed_scales = lzma.compress(bytes(raw_scales), preset=9)
    architecture = _architecture(source)
    header = HEADER.pack(
        MAGIC, architecture["vocab_size"], architecture["hidden_size"], architecture["experts"],
        architecture["heads"], architecture["norm_eps"], len(raw_codes), len(compressed_codes),
        len(raw_scales), len(compressed_scales),
    )
    digest = _digest(header, compressed_codes, compressed_scales)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + digest + compressed_codes + compressed_scales)
    return {
        "format": FORMAT,
        "layout": "complete-scalar-group-int2-fp8-scales-lzma-binary-bitplane-zlib-codes-research",
        "architecture": architecture,
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "scale_type": "float8_e4m3fn-lzma",
        "raw_code_bytes": len(raw_codes),
        "compressed_code_bytes": len(compressed_codes),
        "raw_scale_bytes": len(raw_scales),
        "compressed_scale_bytes": len(compressed_scales),
        "integrity_sha256": digest.hex(),
    }


def _load(path: str | Path) -> tuple[dict, bytes, bytes]:
    payload = Path(path).read_bytes()
    if len(payload) < HEADER.size + DIGEST_BYTES:
        raise ValueError("not a Crystal-9 binary bitplane-code packed INT2 artifact")
    magic, vocab_size, hidden_size, experts, heads, norm_eps, raw_code_size, compressed_code_size, raw_scale_size, compressed_scale_size = HEADER.unpack(payload[:HEADER.size])
    expected_size = HEADER.size + DIGEST_BYTES + compressed_code_size + compressed_scale_size
    if magic != MAGIC or len(payload) != expected_size:
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
    raw_codes = _bitplane_decode(zlib.decompress(compressed_codes), raw_code_size)
    return {"architecture": {"vocab_size": vocab_size, "hidden_size": hidden_size, "experts": experts, "heads": heads, "norm_eps": norm_eps}, "raw_scale_bytes": raw_scale_size}, raw_codes, compressed_scales


class PackedInt2Fp8ScaleLzmaBinaryBitplaneZlibCodesPolicy:
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
    def load(cls, path: str | Path) -> "PackedInt2Fp8ScaleLzmaBinaryBitplaneZlibCodesPolicy":
        return cls(*_load(path))

    def eval(self) -> "PackedInt2Fp8ScaleLzmaBinaryBitplaneZlibCodesPolicy":
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._runtime(token_ids)
