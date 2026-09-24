"""Packed INT2 candidate with one FP8 scale for each two-value group."""

from __future__ import annotations

from pathlib import Path

import torch

from crystal9 import TinyMoEPolicy
from packed_int2 import FORMAT, PackedInt2Policy, _integrity_digest, pack_signed_int2, unpack_signed_int2

GROUP_SIZE = 2
FORMAT_GROUP2 = "crystal-9-packed-int2-group2-scales-v1"


def _encode_group2(values: torch.Tensor) -> dict:
    values = values.detach().cpu().float().contiguous()
    flat = values.reshape(-1)
    padded = torch.nn.functional.pad(flat, (0, (-flat.numel()) % GROUP_SIZE))
    scales = padded.reshape(-1, GROUP_SIZE).abs().amax(dim=1)
    expanded = scales.repeat_interleave(GROUP_SIZE)[: flat.numel()]
    safe = torch.where(expanded == 0, torch.ones_like(expanded), expanded)
    codes = torch.round(flat / safe).clamp(-1, 1).to(torch.int8)
    return {
        "shape": tuple(values.shape),
        "count": flat.numel(),
        "scales": scales.to(torch.float8_e4m3fn),
        "packed": pack_signed_int2(codes),
    }


def export_packed_int2_group2_scales(source: TinyMoEPolicy, path: str | Path) -> dict:
    """Export a distinct shared-scale INT2 hierarchy candidate."""
    tensors = {name: _encode_group2(value) for name, value in source.state_dict().items()}
    manifest = {
        "format": FORMAT_GROUP2,
        "layout": "complete-int2-two-value-shared-fp8-scales-research",
        "architecture": {"vocab_size": source.embedding.num_embeddings, "hidden_size": source.embedding.embedding_dim, "experts": len(source.experts), "heads": source.attention.num_heads, "norm_eps": source.norm.eps},
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "scale_count": sum(record["scales"].numel() for record in tensors.values()),
        "group_size": GROUP_SIZE,
        "scale_type": "float8_e4m3fn",
        "tensors": tensors,
    }
    digest_manifest = dict(manifest)
    digest_manifest["format"] = FORMAT
    manifest["integrity_sha256"] = _integrity_digest(digest_manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, path)
    return {key: value for key, value in manifest.items() if key != "tensors"}


class PackedInt2Group2ScalePolicy(PackedInt2Policy):
    """Independent runtime for the two-value shared-scale artifact."""

    def __init__(self, manifest: dict) -> None:
        if manifest.get("format") != FORMAT_GROUP2:
            raise ValueError("not a Crystal-9 packed INT2 group-2 scale artifact")
        digest_manifest = dict(manifest)
        digest_manifest["format"] = FORMAT
        if manifest.get("integrity_sha256") != _integrity_digest(digest_manifest):
            raise ValueError("packed artifact integrity validation failed")
        self.manifest = manifest
        self.architecture = manifest["architecture"]

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt2Group2ScalePolicy":
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def _tensor(self, name: str, device: torch.device) -> torch.Tensor:
        record = self.manifest["tensors"][name]
        codes = unpack_signed_int2(record["packed"], record["count"]).to(device).float()
        scales = record["scales"].to(device).float().repeat_interleave(GROUP_SIZE)[: record["count"]]
        return (codes * scales).reshape(record["shape"])
