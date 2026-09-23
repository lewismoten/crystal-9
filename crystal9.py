"""Crystal-9 tokenizer and model primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


def quantize_tensor(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric fake quantization for training/evaluation experiments."""
    if bits >= 32:
        return values.clone()
    if bits < 1:
        raise ValueError("bits must be at least 1")
    scale = values.detach().abs().max()
    if scale == 0:
        return values.clone()
    if bits == 1:
        return torch.where(values >= 0, scale, -scale)
    levels = (1 << (bits - 1)) - 1
    return torch.round(values / scale * levels).clamp(-levels, levels) / levels * scale


class TinyMoEPolicy(nn.Module):
    """One-layer causal policy network with a 32-wide routed MoE block."""

    def __init__(self, vocab_size: int, hidden_size: int = 32, experts: int = 9) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position = nn.Embedding(9, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.router = nn.Linear(hidden_size, experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)) for _ in range(experts)]
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position(positions)
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(self.router(state), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        all_experts = torch.stack([expert(state) for expert in self.experts], dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return self.output(state)


@dataclass(frozen=True)
class GameTokenizer:
    tokens: tuple[str, ...]
    input_symbols: frozenset[str]
    max_history_moves: int

    @classmethod
    def from_design_file(cls, path: str | Path) -> "GameTokenizer":
        design = json.loads(Path(path).read_text())
        tokens = tuple(design["vocabulary"]["tokens"])
        return cls(
            tokens=tokens,
            input_symbols=frozenset(design["vocabulary"]["input_symbols"]),
            max_history_moves=design["architecture"]["context_window"],
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode_history(self, history: str) -> list[int]:
        if len(history) > self.max_history_moves:
            raise ValueError(f"history may contain at most {self.max_history_moves} moves")
        unknown = set(history) - self.input_symbols
        if unknown:
            raise ValueError(f"history contains unsupported symbols: {''.join(sorted(unknown))}")
        ids = {token: index for index, token in enumerate(self.tokens)}
        return [ids["<bos>"]] + [ids[symbol] for symbol in history]

    def decode_id(self, token_id: int) -> str:
        return self.tokens[token_id]
