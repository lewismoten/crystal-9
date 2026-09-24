"""Compact binary transport for scalar-FP8 packed INT2 research artifacts."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path

import torch

from crystal9 import TinyMoEPolicy
from packed_int2 import FORMAT as BASE_FORMAT
from packed_int2 import PackedInt2Policy, _encode, _integrity_digest

MAGIC = b"C9I2LZC1"
FORMAT = "crystal-9-packed-int2-fp8-scale-lzma-compact-v1"


def _digest(header: dict, packed: bytes, compressed_scales: bytes) -> str:
    unsigned = {key: value for key, value in header.items() if key != "integrity_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + packed + compressed_scales).hexdigest()


def export_packed_int2_fp8_scale_lzma_compact(source: TinyMoEPolicy, path: str | Path) -> dict:
    """Export scalar-FP8 INT2 codes in a compact, length-delimited binary container."""
    raw_scales = bytearray()
    packed = bytearray()
    tensors = {}
    for name, value in source.state_dict().items():
        record = _encode(value, torch.float8_e4m3fn)
        scale_bytes = record["scales"].contiguous().view(torch.uint8).numpy().tobytes()
        packed_bytes = record["packed"].contiguous().numpy().tobytes()
        tensors[name] = {
            "shape": list(record["shape"]), "count": record["count"],
            "packed_offset": len(packed), "packed_bytes": len(packed_bytes),
            "scale_offset": len(raw_scales), "scale_count": len(scale_bytes),
        }
        packed.extend(packed_bytes)
        raw_scales.extend(scale_bytes)
    compressed_scales = lzma.compress(bytes(raw_scales), preset=9)
    header = {
        "format": FORMAT, "layout": "complete-scalar-group-int2-fp8-scales-lzma-compact-research",
        "architecture": {"vocab_size": source.embedding.num_embeddings, "hidden_size": source.embedding.embedding_dim, "experts": len(source.experts), "heads": source.attention.num_heads, "norm_eps": source.norm.eps},
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "scale_type": "float8_e4m3fn-lzma", "tensors": tensors,
        "raw_scale_bytes": len(raw_scales), "compressed_scale_bytes": len(compressed_scales),
        "packed_bytes": len(packed),
    }
    header["integrity_sha256"] = _digest(header, bytes(packed), compressed_scales)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    payload = MAGIC + len(header_bytes).to_bytes(4, "big") + header_bytes + bytes(packed) + compressed_scales
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return header


def _load(path: str | Path) -> tuple[dict, bytes, bytes]:
    payload = Path(path).read_bytes()
    if payload[:len(MAGIC)] != MAGIC or len(payload) < len(MAGIC) + 4:
        raise ValueError("not a Crystal-9 compact LZMA-FP8 packed INT2 artifact")
    header_length = int.from_bytes(payload[len(MAGIC):len(MAGIC) + 4], "big")
    header_start = len(MAGIC) + 4
    header_end = header_start + header_length
    try:
        header = json.loads(payload[header_start:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid compact packed artifact header") from error
    packed_end = header_end + header.get("packed_bytes", -1)
    if header.get("format") != FORMAT or packed_end > len(payload):
        raise ValueError("invalid compact packed artifact payload")
    packed, compressed_scales = payload[header_end:packed_end], payload[packed_end:]
    if header.get("integrity_sha256") != _digest(header, packed, compressed_scales):
        raise ValueError("packed artifact integrity validation failed")
    return header, packed, compressed_scales


class PackedInt2Fp8ScaleLzmaCompactPolicy:
    """Runtime for the compact lossless LZMA transport."""

    def __init__(self, header: dict, packed: bytes, compressed_scales: bytes) -> None:
        raw_scales = lzma.decompress(compressed_scales)
        if len(raw_scales) != header["raw_scale_bytes"]:
            raise ValueError("compressed scale payload length validation failed")
        tensors = {}
        for name, record in header["tensors"].items():
            packed_start = record["packed_offset"]
            packed_end = packed_start + record["packed_bytes"]
            scale_start = record["scale_offset"]
            scale_end = scale_start + record["scale_count"]
            if packed_end > len(packed) or scale_end > len(raw_scales):
                raise ValueError("compact tensor offset validation failed")
            tensors[name] = {
                "shape": tuple(record["shape"]), "count": record["count"],
                "packed": torch.frombuffer(bytearray(packed[packed_start:packed_end]), dtype=torch.uint8).clone(),
                "scales": torch.frombuffer(bytearray(raw_scales[scale_start:scale_end]), dtype=torch.uint8).view(torch.float8_e4m3fn).clone(),
            }
        manifest = {"format": BASE_FORMAT, "layout": "complete-scalar-group-int2-research", "architecture": header["architecture"], "parameter_values": header["parameter_values"], "scale_type": "float8_e4m3fn", "tensors": tensors}
        manifest["integrity_sha256"] = _integrity_digest(manifest)
        self._runtime = PackedInt2Policy(manifest)

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt2Fp8ScaleLzmaCompactPolicy":
        return cls(*_load(path))

    def eval(self) -> "PackedInt2Fp8ScaleLzmaCompactPolicy":
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._runtime(token_ids)
