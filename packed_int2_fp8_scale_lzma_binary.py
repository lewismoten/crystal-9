"""Binary-metadata transport for scalar-FP8 packed INT2 research artifacts."""

from __future__ import annotations

import hashlib
import lzma
import struct
from pathlib import Path

import torch

from crystal9 import TinyMoEPolicy
from packed_int2 import FORMAT as BASE_FORMAT
from packed_int2 import PackedInt2Policy, _encode, _integrity_digest

MAGIC = b"C9I2LZB1"
FORMAT = "crystal-9-packed-int2-fp8-scale-lzma-binary-v1"
HEADER = struct.Struct(">8sIIIIfIII")
DIGEST_BYTES = 32


def _digest(header: bytes, packed: bytes, compressed_scales: bytes) -> bytes:
    return hashlib.sha256(header + packed + compressed_scales).digest()


def _architecture(source: TinyMoEPolicy) -> dict:
    return {
        "vocab_size": source.embedding.num_embeddings,
        "hidden_size": source.embedding.embedding_dim,
        "experts": len(source.experts),
        "heads": source.attention.num_heads,
        "norm_eps": source.norm.eps,
    }


def export_packed_int2_fp8_scale_lzma_binary(source: TinyMoEPolicy, path: str | Path) -> dict:
    """Export canonical scalar-FP8 INT2 values with binary canonical-layout metadata."""
    raw_scales = bytearray()
    packed = bytearray()
    for value in source.state_dict().values():
        record = _encode(value, torch.float8_e4m3fn)
        packed.extend(record["packed"].contiguous().numpy().tobytes())
        raw_scales.extend(record["scales"].contiguous().view(torch.uint8).numpy().tobytes())
    compressed_scales = lzma.compress(bytes(raw_scales), preset=9)
    architecture = _architecture(source)
    header = HEADER.pack(
        MAGIC, architecture["vocab_size"], architecture["hidden_size"],
        architecture["experts"], architecture["heads"], architecture["norm_eps"],
        len(packed), len(raw_scales), len(compressed_scales),
    )
    digest = _digest(header, bytes(packed), compressed_scales)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + digest + bytes(packed) + compressed_scales)
    return {
        "format": FORMAT,
        "layout": "complete-scalar-group-int2-fp8-scales-lzma-binary-research",
        "architecture": architecture,
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "scale_type": "float8_e4m3fn-lzma",
        "raw_scale_bytes": len(raw_scales),
        "compressed_scale_bytes": len(compressed_scales),
        "packed_bytes": len(packed),
        "integrity_sha256": digest.hex(),
    }


def _load(path: str | Path) -> tuple[dict, bytes, bytes]:
    payload = Path(path).read_bytes()
    if len(payload) < HEADER.size + DIGEST_BYTES:
        raise ValueError("not a Crystal-9 binary LZMA-FP8 packed INT2 artifact")
    magic, vocab_size, hidden_size, experts, heads, norm_eps, packed_size, raw_scale_size, compressed_size = HEADER.unpack(payload[:HEADER.size])
    if magic != MAGIC or len(payload) != HEADER.size + DIGEST_BYTES + packed_size + compressed_size:
        raise ValueError("invalid binary packed artifact payload")
    digest_start = HEADER.size
    packed_start = digest_start + DIGEST_BYTES
    packed_end = packed_start + packed_size
    digest = payload[digest_start:packed_start]
    header, packed, compressed_scales = payload[:HEADER.size], payload[packed_start:packed_end], payload[packed_end:]
    if digest != _digest(header, packed, compressed_scales):
        raise ValueError("packed artifact integrity validation failed")
    return {
        "architecture": {"vocab_size": vocab_size, "hidden_size": hidden_size, "experts": experts, "heads": heads, "norm_eps": norm_eps},
        "raw_scale_bytes": raw_scale_size,
    }, packed, compressed_scales


class PackedInt2Fp8ScaleLzmaBinaryPolicy:
    """Runtime for canonical-layout scalar FP8 scales with binary metadata."""

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
    def load(cls, path: str | Path) -> "PackedInt2Fp8ScaleLzmaBinaryPolicy":
        return cls(*_load(path))

    def eval(self) -> "PackedInt2Fp8ScaleLzmaBinaryPolicy":
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._runtime(token_ids)
