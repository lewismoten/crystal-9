"""Packed INT3 export and independent runtime for the accepted complete Crystal-9 layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from crystal9 import TinyMoEPolicy

FORMAT = "crystal-9-packed-int3-v1"
LEVELS = 3
WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 4, 6), (0, 4, 8), (2, 4, 6))


def pack_signed_int3(values: torch.Tensor) -> torch.Tensor:
    """Pack signed [-3, 3] values low-bit first, padding to full bytes."""
    if values.dtype != torch.int8 or values.ndim != 1 or torch.any((values < -3) | (values > 3)):
        raise ValueError("INT3 packing requires a rank-1 [-3, 3] torch.int8 tensor")
    unsigned = (values.to(torch.int16) & 0x7).tolist()
    packed = torch.zeros((len(unsigned) * 3 + 7) // 8, dtype=torch.uint8)
    for index, value in enumerate(unsigned):
        bit = index * 3
        byte = bit // 8
        shift = bit % 8
        packed[byte] |= value << shift
        if shift > 5:
            packed[byte + 1] |= value >> (8 - shift)
    return packed


def unpack_signed_int3(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack low-bit-first signed INT3 values, discarding padding."""
    if packed.dtype != torch.uint8 or packed.ndim != 1 or count < 0 or count * 3 > packed.numel() * 8:
        raise ValueError("invalid packed INT3 buffer or count")
    values = []
    raw = packed.tolist()
    for index in range(count):
        bit = index * 3
        byte = bit // 8
        shift = bit % 8
        value = raw[byte] >> shift
        if shift > 5:
            value |= raw[byte + 1] << (8 - shift)
        value &= 0x7
        values.append(value - 8 if value & 0x4 else value)
    return torch.tensor(values, dtype=torch.int8)


def _encode(values: torch.Tensor, scheme: str, group_size: int | None = None) -> dict:
    values = values.detach().cpu().float().contiguous()
    if scheme == "row":
        groups = values.reshape(values.shape[0], -1)
    elif scheme == "group":
        if group_size is None or values.numel() % group_size:
            raise ValueError("group size must divide tensor length")
        groups = values.reshape(-1, group_size)
    elif scheme == "tensor":
        groups = values.reshape(1, -1)
    else:
        raise ValueError(f"unknown INT3 scheme: {scheme}")
    scales = groups.abs().amax(dim=1)
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(groups / safe_scales.unsqueeze(1) * LEVELS).clamp(-LEVELS, LEVELS).to(torch.int8)
    return {
        "shape": tuple(values.shape), "scheme": scheme, "group_size": group_size,
        "count": values.numel(), "scales": scales, "packed": pack_signed_int3(codes.reshape(-1)),
    }


def _decode(record: dict, device: torch.device) -> torch.Tensor:
    codes = unpack_signed_int3(record["packed"], record["count"]).to(device).float()
    if record["scheme"] == "row":
        groups = codes.reshape(record["shape"][0], -1)
    elif record["scheme"] == "group":
        groups = codes.reshape(-1, record["group_size"])
    else:
        groups = codes.reshape(1, -1)
    return (groups / LEVELS * record["scales"].to(device).reshape(-1, 1)).reshape(record["shape"])


def _record_digest(digest, name: str, record: dict) -> None:
    metadata = {key: record[key] for key in ("shape", "scheme", "group_size", "count")}
    digest.update(json.dumps({"name": name, **metadata}, sort_keys=True, separators=(",", ":")).encode())
    digest.update(record["scales"].contiguous().numpy().tobytes())
    digest.update(record["packed"].contiguous().numpy().tobytes())


def _integrity_digest(manifest: dict) -> str:
    digest = hashlib.sha256()
    header = {key: manifest[key] for key in ("format", "architecture", "parameter_values", "layout")}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(manifest["tensors"]):
        record = manifest["tensors"][name]
        if "segments" in record:
            digest.update(name.encode())
            for segment, segment_record in sorted(record["segments"].items()):
                _record_digest(digest, segment, segment_record)
        else:
            _record_digest(digest, name, record)
    return digest.hexdigest()


def _valid_live_history(history: str, symbols: frozenset[str], max_moves: int) -> bool:
    if not isinstance(history, str) or len(history) > max_moves or set(history) - symbols:
        return False
    board = ["."] * 9
    for index, symbol in enumerate(history):
        if any(board[a] != "." and board[a] == board[b] == board[c] for a, b, c in WINS):
            return False
        square = "abcdefghi".index(symbol)
        if board[square] != ".":
            return False
        board[square] = "X" if index % 2 == 0 else "O"
    return not any(board[a] != "." and board[a] == board[b] == board[c] for a, b, c in WINS) and "." in board


def _scheme_for(name: str, value: torch.Tensor) -> tuple[str, int | None]:
    if name == "router.weight":
        return "group", 4
    if name in {"embedding.weight", "position.weight", "attention.out_proj.weight", "attention.in_proj_bias", "attention.out_proj.bias", "norm.weight", "norm.bias"}:
        return "group", 1
    if name == "attention.in_proj_weight":
        raise ValueError("attention input projection is segmented")
    if name.endswith(".weight"):
        return "row", None
    return "tensor", None


def export_packed_int3(source: TinyMoEPolicy, path: str | Path) -> dict:
    """Export the accepted complete mixed-layout INT3 model as genuine 3-bit codes."""
    tensors = {}
    for name, value in source.state_dict().items():
        if name == "attention.in_proj_weight":
            width = source.attention.embed_dim
            tensors[name] = {"segments": {
                "q": _encode(value[:width], "row"),
                "k": _encode(value[width:2 * width], "group", 2),
                "v": _encode(value[2 * width:], "group", 2),
            }}
        else:
            scheme, group_size = _scheme_for(name, value)
            tensors[name] = _encode(value, scheme, group_size)
    manifest = {
        "format": FORMAT,
        "layout": "complete-mixed-int3-scope14",
        "architecture": {"vocab_size": source.embedding.num_embeddings, "hidden_size": source.embedding.embedding_dim, "experts": len(source.experts), "heads": source.attention.num_heads, "norm_eps": source.norm.eps},
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "tensors": tensors,
    }
    manifest["integrity_sha256"] = _integrity_digest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, path)
    return {key: value for key, value in manifest.items() if key != "tensors"}


class PackedInt3Policy:
    """Independent runtime consuming only a complete packed INT3 artifact."""

    def __init__(self, manifest: dict) -> None:
        if manifest.get("format") != FORMAT:
            raise ValueError("not a Crystal-9 packed INT3 artifact")
        if manifest.get("integrity_sha256") != _integrity_digest(manifest):
            raise ValueError("packed artifact integrity validation failed")
        self.manifest = manifest
        self.architecture = manifest["architecture"]

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt3Policy":
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def eval(self) -> "PackedInt3Policy":
        return self

    def _tensor(self, name: str, device: torch.device) -> torch.Tensor:
        record = self.manifest["tensors"][name]
        if "segments" not in record:
            return _decode(record, device)
        return torch.cat([_decode(record["segments"][segment], device) for segment in ("q", "k", "v")])

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

    def predict(self, history: str, tokenizer) -> str:
        if not _valid_live_history(history, tokenizer.input_symbols, tokenizer.max_history_moves):
            return "!"
        encoded = tokenizer.encode_history(history)
        token_ids = torch.tensor([encoded + [0] * (9 - len(encoded))])
        return tokenizer.decode_id(self(token_ids).argmax(dim=-1).item())


def evaluate_packed(runtime: PackedInt3Policy, tokenizer, device: torch.device, histories: list[str], expected_move) -> dict[str, int]:
    misses = 0
    with torch.no_grad():
        for start in range(0, len(histories), 4096):
            batch = histories[start:start + 4096]
            inputs = torch.tensor([encoded + [0] * (9 - len(encoded)) for history in batch for encoded in [tokenizer.encode_history(history)]], device=device)
            predicted = runtime(inputs).argmax(dim=-1).tolist()
            misses += sum(tokenizer.decode_id(token) != expected_move(history) for token, history in zip(predicted, batch))
    return {"legal_histories": len(histories), "policy_misses": misses}
