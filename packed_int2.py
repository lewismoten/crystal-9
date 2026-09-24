"""Packed scalar-group INT2 export and independent runtime for Crystal-9 research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from crystal9 import TinyMoEPolicy

FORMAT = "crystal-9-packed-int2-v1"
LEVELS = 1
WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def pack_signed_int2(values: torch.Tensor) -> torch.Tensor:
    """Pack signed scalar-INT2 [-1, 1] codes low-bit first."""
    if values.dtype != torch.int8 or values.ndim != 1 or torch.any((values < -1) | (values > 1)):
        raise ValueError("INT2 packing requires a rank-1 [-1, 1] torch.int8 tensor")
    codes = (values.to(torch.int16) & 0x3).to(torch.uint8)
    if codes.numel() % 4:
        codes = torch.cat((codes, torch.zeros(4 - codes.numel() % 4, dtype=torch.uint8)))
    return (codes[::4] | (codes[1::4] << 2) | (codes[2::4] << 4) | (codes[3::4] << 6)).contiguous()


def unpack_signed_int2(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack low-bit-first signed INT2 codes, discarding padding."""
    if packed.dtype != torch.uint8 or packed.ndim != 1 or count < 0 or count > packed.numel() * 4:
        raise ValueError("invalid packed INT2 buffer or count")
    shifts = torch.arange(4, dtype=torch.uint8).mul(2)
    raw = ((packed.unsqueeze(1) >> shifts) & 0x3).reshape(-1)[:count]
    return raw.to(torch.int8).sub_(torch.where(raw >= 2, 4, 0).to(torch.int8))


def _encode(values: torch.Tensor, scale_dtype: torch.dtype = torch.float32) -> dict:
    values = values.detach().cpu().float().contiguous()
    scales = values.reshape(-1).abs()
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(values.reshape(-1) / safe_scales * LEVELS).clamp(-LEVELS, LEVELS).to(torch.int8)
    return {"shape": tuple(values.shape), "count": values.numel(), "scales": scales.to(scale_dtype), "packed": pack_signed_int2(codes)}


def _decode(record: dict, device: torch.device) -> torch.Tensor:
    codes = unpack_signed_int2(record["packed"], record["count"]).to(device).float()
    return (codes / LEVELS * record["scales"].to(device).float()).reshape(record["shape"])


def _integrity_digest(manifest: dict) -> str:
    digest = hashlib.sha256()
    header = {key: manifest[key] for key in ("format", "architecture", "parameter_values", "layout")}
    if "scale_type" in manifest:
        header["scale_type"] = manifest["scale_type"]
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(manifest["tensors"]):
        record = manifest["tensors"][name]
        digest.update(json.dumps({"name": name, "shape": record["shape"], "count": record["count"]}, sort_keys=True, separators=(",", ":")).encode())
        digest.update(record["scales"].contiguous().view(torch.uint8).numpy().tobytes())
        digest.update(record["packed"].contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_packed_int2(source: TinyMoEPolicy, path: str | Path, scale_dtype: torch.dtype = torch.float32) -> dict:
    """Export the full scalar-group INT2 proof as genuine packed 2-bit codes."""
    manifest = {
        "format": FORMAT,
        "layout": "complete-scalar-group-int2-research",
        "architecture": {"vocab_size": source.embedding.num_embeddings, "hidden_size": source.embedding.embedding_dim, "experts": len(source.experts), "heads": source.attention.num_heads, "norm_eps": source.norm.eps},
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "scale_type": str(scale_dtype).removeprefix("torch."),
        "tensors": {name: _encode(value, scale_dtype) for name, value in source.state_dict().items()},
    }
    manifest["integrity_sha256"] = _integrity_digest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, path)
    return {key: value for key, value in manifest.items() if key != "tensors"}


class PackedInt2Policy:
    """Independent runtime consuming only a complete packed scalar-group INT2 artifact."""

    def __init__(self, manifest: dict) -> None:
        if manifest.get("format") != FORMAT:
            raise ValueError("not a Crystal-9 packed INT2 artifact")
        if manifest.get("integrity_sha256") != _integrity_digest(manifest):
            raise ValueError("packed artifact integrity validation failed")
        self.manifest = manifest
        self.architecture = manifest["architecture"]

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt2Policy":
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def eval(self) -> "PackedInt2Policy":
        return self

    def _tensor(self, name: str, device: torch.device) -> torch.Tensor:
        return _decode(self.manifest["tensors"][name], device)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        device = token_ids.device
        t = lambda name: self._tensor(name, device)
        batch, steps = token_ids.shape
        width = self.architecture["hidden_size"]
        positions = torch.arange(steps, device=device).unsqueeze(0)
        hidden = F.embedding(token_ids, t("embedding.weight"), padding_idx=0) + F.embedding(positions, t("position.weight"))
        qkv = F.linear(hidden, t("attention.in_proj_weight"), t("attention.in_proj_bias"))
        query, key, value = qkv.chunk(3, dim=-1)
        heads = self.architecture["heads"]
        head_width = width // heads
        query, key, value = (part.view(batch, steps, heads, head_width).transpose(1, 2) for part in (query, key, value))
        scores = (query @ key.transpose(-2, -1)) * (head_width ** -0.5)
        causal = torch.triu(torch.ones(steps, steps, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf")).masked_fill(token_ids.eq(0).view(batch, 1, 1, steps), float("-inf"))
        attended = (torch.softmax(scores, dim=-1) @ value).transpose(1, 2).contiguous().view(batch, steps, width)
        attended = F.linear(attended, t("attention.out_proj.weight"), t("attention.out_proj.bias"))
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = F.layer_norm(attended[torch.arange(batch, device=device), last], (width,), t("norm.weight"), t("norm.bias"), self.architecture["norm_eps"])
        routing = torch.softmax(F.linear(state, t("router.weight"), t("router.bias")), dim=-1)
        top_weights, top_indices = routing.topk(2, dim=-1)
        experts = []
        for index in range(self.architecture["experts"]):
            value = F.silu(F.linear(state, t(f"experts.{index}.0.weight"), t(f"experts.{index}.0.bias")))
            experts.append(F.linear(value, t(f"experts.{index}.2.weight"), t(f"experts.{index}.2.bias")))
        routed = torch.stack(experts, dim=1).gather(1, top_indices.unsqueeze(-1).expand(-1, -1, width))
        return F.linear(state + (routed * top_weights.unsqueeze(-1)).sum(dim=1), t("output.weight"), t("output.bias"))


def evaluate_packed(runtime, tokenizer, device: torch.device, histories: list[str], expected_move) -> dict[str, int]:
    """Exhaustively score a packed runtime against the authoritative policy."""
    misses = 0
    with torch.no_grad():
        for start in range(0, len(histories), 4096):
            batch = histories[start:start + 4096]
            inputs = torch.tensor(
                [tokenizer.encode_history(history) + [0] * (9 - len(tokenizer.encode_history(history))) for history in batch],
                device=device,
            )
            predicted = runtime(inputs).argmax(dim=-1).tolist()
            misses += sum(tokenizer.decode_id(token) != expected_move(history) for token, history in zip(predicted, batch))
    return {"legal_histories": len(histories), "policy_misses": misses}
