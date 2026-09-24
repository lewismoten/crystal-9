"""Lossless Zstandard scale transport for the scalar-group INT2 FP8 research layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import torch

from crystal9 import TinyMoEPolicy
from packed_int2 import FORMAT as BASE_FORMAT
from packed_int2 import PackedInt2Policy, _encode, _integrity_digest

FORMAT = "crystal-9-packed-int2-fp8-scale-zstd-v1"


def _zstd(arguments: list[str], payload: bytes) -> bytes:
    result = subprocess.run(["zstd", "-q", *arguments, "--stdout"], input=payload, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or "zstd failed")
    return result.stdout


def _digest(manifest: dict) -> str:
    digest = hashlib.sha256()
    header = {key: manifest[key] for key in ("format", "architecture", "parameter_values", "layout", "scale_type")}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(manifest["tensors"]):
        record = manifest["tensors"][name]
        digest.update(json.dumps({"name": name, "shape": record["shape"], "count": record["count"], "scale_offset": record["scale_offset"], "scale_count": record["scale_count"]}, sort_keys=True, separators=(",", ":")).encode())
        digest.update(record["packed"].contiguous().numpy().tobytes())
    digest.update(manifest["compressed_scales"])
    return digest.hexdigest()


def export_packed_int2_fp8_scale_zstd(source: TinyMoEPolicy, path: str | Path) -> dict:
    """Export packed INT2 codes with losslessly Zstandard-compressed FP8 scalar scales."""
    raw_scales = bytearray()
    tensors = {}
    for name, value in source.state_dict().items():
        record = _encode(value, torch.float8_e4m3fn)
        raw = record.pop("scales").contiguous().view(torch.uint8).numpy().tobytes()
        tensors[name] = {**record, "scale_offset": len(raw_scales), "scale_count": len(raw)}
        raw_scales.extend(raw)
    compressed_scales = _zstd(["-19"], bytes(raw_scales))
    manifest = {"format": FORMAT, "layout": "complete-scalar-group-int2-fp8-scales-zstd-research", "architecture": {"vocab_size": source.embedding.num_embeddings, "hidden_size": source.embedding.embedding_dim, "experts": len(source.experts), "heads": source.attention.num_heads, "norm_eps": source.norm.eps}, "parameter_values": sum(value.numel() for value in source.parameters()), "scale_type": "float8_e4m3fn-zstd", "tensors": tensors, "raw_scale_bytes": len(raw_scales), "compressed_scale_bytes": len(compressed_scales), "compressed_scales": compressed_scales}
    manifest["integrity_sha256"] = _digest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, path)
    return {key: value for key, value in manifest.items() if key not in {"tensors", "compressed_scales"}}


class PackedInt2Fp8ScaleZstdPolicy:
    """Runtime which validates and losslessly expands Zstandard-compressed FP8 scales."""

    def __init__(self, manifest: dict) -> None:
        if manifest.get("format") != FORMAT:
            raise ValueError("not a Crystal-9 Zstandard-compressed-FP8 packed INT2 artifact")
        if manifest.get("integrity_sha256") != _digest(manifest):
            raise ValueError("packed artifact integrity validation failed")
        raw_scales = _zstd(["-d"], manifest["compressed_scales"])
        if len(raw_scales) != manifest["raw_scale_bytes"]:
            raise ValueError("compressed scale payload length validation failed")
        tensors = {}
        for name, record in manifest["tensors"].items():
            start = record["scale_offset"]
            end = start + record["scale_count"]
            scales = torch.frombuffer(bytearray(raw_scales[start:end]), dtype=torch.uint8).view(torch.float8_e4m3fn).clone()
            tensors[name] = {key: value for key, value in record.items() if key not in {"scale_offset", "scale_count"}}
            tensors[name]["scales"] = scales
        base_manifest = {"format": BASE_FORMAT, "layout": "complete-scalar-group-int2-research", "architecture": manifest["architecture"], "parameter_values": manifest["parameter_values"], "scale_type": "float8_e4m3fn", "tensors": tensors}
        base_manifest["integrity_sha256"] = _integrity_digest(base_manifest)
        self._runtime = PackedInt2Policy(base_manifest)

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt2Fp8ScaleZstdPolicy":
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def eval(self) -> "PackedInt2Fp8ScaleZstdPolicy":
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._runtime(token_ids)
