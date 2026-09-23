"""Packed INT4 export and independent runtime for Crystal-9."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from crystal9 import TinyMoEPolicy, pack_signed_int4, unpack_signed_int4

FORMAT = "crystal-9-packed-int4-v1"
LEVELS = 7
WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def _winner(board: list[str]) -> str | None:
    for a, b, c in WINS:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None


def _valid_live_history(history: str, symbols: frozenset[str], max_moves: int) -> bool:
    if not isinstance(history, str) or len(history) > max_moves or set(history) - symbols:
        return False
    board = ["."] * 9
    ordered_symbols = "abcdefghi"
    for index, symbol in enumerate(history):
        if _winner(board):
            return False
        square = ordered_symbols.index(symbol)
        if board[square] != ".":
            return False
        board[square] = "X" if index % 2 == 0 else "O"
    return _winner(board) is None and "." in board


def _integrity_digest(manifest: dict) -> str:
    """Hash the complete packed tensor inventory and format-critical metadata."""
    digest = hashlib.sha256()
    header = {
        "format": manifest["format"],
        "architecture": manifest["architecture"],
        "norm_weight_group_size": manifest["norm_weight_group_size"],
        "parameter_values": manifest["parameter_values"],
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(manifest["tensors"]):
        record = manifest["tensors"][name]
        metadata = {
            "name": name,
            "shape": list(record["shape"]),
            "scheme": record["scheme"],
            "group_size": record["group_size"],
            "count": record["count"],
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        digest.update(record["scales"].detach().cpu().float().contiguous().numpy().tobytes())
        digest.update(record["packed"].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _encode(values: torch.Tensor, scheme: str, group_size: int | None = None) -> dict:
    values = values.detach().cpu().float().contiguous()
    if scheme == "row":
        groups = values.reshape(values.shape[0], -1)
    elif scheme == "group":
        if group_size is None or values.numel() % group_size:
            raise ValueError("group size must divide the tensor length")
        groups = values.reshape(-1, group_size)
    elif scheme == "tensor":
        groups = values.reshape(1, -1)
    else:
        raise ValueError(f"unknown INT4 scheme: {scheme}")
    scales = groups.abs().amax(dim=1)
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(groups / safe_scales.unsqueeze(1) * LEVELS).clamp(-LEVELS, LEVELS).to(torch.int8)
    return {
        "shape": tuple(values.shape),
        "scheme": scheme,
        "group_size": group_size,
        "count": values.numel(),
        "scales": scales,
        "packed": pack_signed_int4(codes.reshape(-1)),
    }


def _decode(record: dict, device: torch.device) -> torch.Tensor:
    codes = unpack_signed_int4(record["packed"].to(device), record["count"]).float()
    if record["scheme"] == "row":
        groups = codes.reshape(record["shape"][0], -1)
    elif record["scheme"] == "group":
        groups = codes.reshape(-1, record["group_size"])
    else:
        groups = codes.reshape(1, -1)
    return (groups * record["scales"].to(device).reshape(-1, 1) / LEVELS).reshape(record["shape"])


def _scheme_for(name: str, value: torch.Tensor, norm_weight_group_size: int) -> tuple[str, int | None]:
    if name == "norm.weight":
        return "group", norm_weight_group_size
    if value.ndim == 2:
        return "row", None
    if value.ndim == 1:
        return "tensor", None
    raise ValueError(f"unsupported parameter rank for {name}: {value.ndim}")


def export_packed_int4(source: TinyMoEPolicy, path: str | Path, norm_weight_group_size: int = 2) -> dict:
    """Export every Crystal-9 parameter as packed signed INT4 plus explicit scales."""
    if norm_weight_group_size < 1 or source.norm.weight.numel() % norm_weight_group_size:
        raise ValueError("invalid LayerNorm weight group size")
    tensors = {}
    for name, value in source.state_dict().items():
        scheme, group_size = _scheme_for(name, value, norm_weight_group_size)
        tensors[name] = _encode(value, scheme, group_size)
    manifest = {
        "format": FORMAT,
        "architecture": {"vocab_size": source.embedding.num_embeddings, "hidden_size": source.embedding.embedding_dim, "experts": len(source.experts), "heads": source.attention.num_heads, "norm_eps": source.norm.eps},
        "norm_weight_group_size": norm_weight_group_size,
        "parameter_values": sum(value.numel() for value in source.parameters()),
        "tensors": tensors,
    }
    manifest["integrity_sha256"] = _integrity_digest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, path)
    return {key: value for key, value in manifest.items() if key != "tensors"}


class PackedInt4Policy:
    """Independent inference runtime consuming only a packed Crystal-9 artifact."""

    def __init__(self, manifest: dict) -> None:
        if manifest.get("format") != FORMAT:
            raise ValueError("not a Crystal-9 packed INT4 artifact")
        if manifest.get("integrity_sha256") != _integrity_digest(manifest):
            raise ValueError("packed artifact integrity validation failed")
        self.manifest = manifest
        self.architecture = manifest["architecture"]

    @classmethod
    def load(cls, path: str | Path) -> "PackedInt4Policy":
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def eval(self) -> "PackedInt4Policy":
        return self

    def _tensor(self, name: str, device: torch.device) -> torch.Tensor:
        return _decode(self.manifest["tensors"][name], device)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        device = token_ids.device
        t = lambda name: self._tensor(name, device)
        batch, steps = token_ids.shape
        width = self.architecture["hidden_size"]
        positions = torch.arange(steps, device=device).unsqueeze(0)
        hidden = F.embedding(token_ids, t("embedding.weight"), padding_idx=0)
        hidden = hidden + F.embedding(positions, t("position.weight"))

        qkv = F.linear(hidden, t("attention.in_proj_weight"), t("attention.in_proj_bias"))
        query, key, value = qkv.chunk(3, dim=-1)
        heads = self.architecture["heads"]
        head_width = width // heads
        query = query.view(batch, steps, heads, head_width).transpose(1, 2)
        key = key.view(batch, steps, heads, head_width).transpose(1, 2)
        value = value.view(batch, steps, heads, head_width).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) * (head_width ** -0.5)
        causal = torch.triu(torch.ones(steps, steps, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        scores = scores.masked_fill(token_ids.eq(0).view(batch, 1, 1, steps), float("-inf"))
        attended = torch.softmax(scores, dim=-1) @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, steps, width)
        attended = F.linear(attended, t("attention.out_proj.weight"), t("attention.out_proj.bias"))

        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        selected = attended[torch.arange(batch, device=device), last]
        state = F.layer_norm(selected, (width,), t("norm.weight"), t("norm.bias"), self.architecture["norm_eps"])
        routing = torch.softmax(F.linear(state, t("router.weight"), t("router.bias")), dim=-1)
        top_weights, top_indices = routing.topk(2, dim=-1)
        expert_outputs = []
        for index in range(self.architecture["experts"]):
            value = F.linear(state, t(f"experts.{index}.0.weight"), t(f"experts.{index}.0.bias"))
            value = F.silu(value)
            expert_outputs.append(F.linear(value, t(f"experts.{index}.2.weight"), t(f"experts.{index}.2.bias")))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, width))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, t("output.weight"), t("output.bias"))

    def predict(self, history: str, tokenizer) -> str:
        """Return exactly one legal policy token, or ``!`` for an invalid history."""
        if not _valid_live_history(history, tokenizer.input_symbols, tokenizer.max_history_moves):
            return "!"
        encoded = tokenizer.encode_history(history)
        device = torch.device("cpu")
        token_ids = torch.tensor([encoded + [0] * (9 - len(encoded))], device=device)
        return tokenizer.decode_id(self(token_ids).argmax(dim=-1).item())


def evaluate_packed(runtime: PackedInt4Policy, tokenizer, device: torch.device, histories: list[str], expected_move) -> dict[str, int]:
    """Evaluate a packed runtime directly, without a PyTorch model checkpoint."""
    misses = 0
    runtime.eval()
    with torch.no_grad():
        for start in range(0, len(histories), 4096):
            batch = histories[start : start + 4096]
            inputs = torch.tensor(
                [encoded + [0] * (9 - len(encoded)) for history in batch for encoded in [tokenizer.encode_history(history)]],
                device=device,
            )
            predicted = runtime(inputs).argmax(dim=-1).tolist()
            misses += sum(tokenizer.decode_id(token_id) != expected_move(history) for token_id, history in zip(predicted, batch))
    return {"legal_histories": len(histories), "policy_misses": misses}
