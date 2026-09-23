"""Crystal-9 tokenizer and model primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


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


def quantize_ste(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Fake quantize in the forward pass while preserving weight gradients."""
    quantized = quantize_tensor(values, bits)
    return values + (quantized - values).detach()


def quantize_rows(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetrically quantize every output row with its own scale."""
    if values.ndim != 2:
        raise ValueError("row quantization requires a rank-2 tensor")
    if bits >= 32:
        return values.clone()
    levels = (1 << (bits - 1)) - 1
    scale = values.detach().abs().amax(dim=1, keepdim=True)
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(values / safe_scale * levels).clamp(-levels, levels) / levels * safe_scale
    return torch.where(scale == 0, values, quantized)


def quantize_rows_ste(values: torch.Tensor, bits: int) -> torch.Tensor:
    quantized = quantize_rows(values, bits)
    return values + (quantized - values).detach()


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

    def forward_mixed_int4(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The exact mixed layout used for INT4 QAT and materialized gating.

        Only routed-expert and output weight matrices use INT4 row quantization.
        Attention, embeddings, router, norms, and all biases remain F32.
        """
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position(positions)
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(self.router(state), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 4), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 4), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 4), self.output.bias)

    def forward_mixed_int4_input(self, token_ids: torch.Tensor) -> torch.Tensor:
        """INT4 row-quantize input tables; retain the proven INT4 suffix."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = F.embedding(token_ids, quantize_rows_ste(self.embedding.weight, 4), padding_idx=0)
        hidden = hidden + F.embedding(positions, quantize_rows_ste(self.position.weight, 4))
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(self.router(state), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 4), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 4), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 4), self.output.bias)

    def forward_quantized(self, token_ids: torch.Tensor, bits: int) -> torch.Tensor:
        """QAT forward path for the embedding, routed MLP, router, and head.

        Multihead attention remains F32 in this first QAT stage; it is a
        separate packed-runtime concern and not silently represented as int4.
        """
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = F.embedding(token_ids, quantize_ste(self.embedding.weight, bits), padding_idx=0)
        hidden = hidden + F.embedding(positions, quantize_ste(self.position.weight, bits))
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = F.layer_norm(
            attended[torch.arange(token_ids.shape[0], device=token_ids.device), last],
            self.norm.normalized_shape,
            quantize_ste(self.norm.weight, bits),
            quantize_ste(self.norm.bias, bits),
            self.norm.eps,
        )
        router_weights = torch.softmax(F.linear(state, quantize_ste(self.router.weight, bits), quantize_ste(self.router.bias, bits)), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_ste(first.weight, bits), quantize_ste(first.bias, bits))
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_ste(second.weight, bits), quantize_ste(second.bias, bits)))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_ste(self.output.weight, bits), quantize_ste(self.output.bias, bits))


def materialize_mixed_int4(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Copy the exact INT4-row tensors used by ``forward_mixed_int4``."""
    materialized = TinyMoEPolicy(
        source.embedding.num_embeddings,
        source.embedding.embedding_dim,
        len(source.experts),
    ).to(next(source.parameters()).device)
    materialized.load_state_dict(source.state_dict())
    with torch.no_grad():
        for expert in materialized.experts:
            expert[0].weight.copy_(quantize_rows(expert[0].weight, 4))
            expert[2].weight.copy_(quantize_rows(expert[2].weight, 4))
        materialized.output.weight.copy_(quantize_rows(materialized.output.weight, 4))
    return materialized


def materialize_mixed_int4_input(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the input-table stage plus the frozen mixed-INT4 suffix."""
    materialized = materialize_mixed_int4(source)
    with torch.no_grad():
        materialized.embedding.weight.copy_(quantize_rows(materialized.embedding.weight, 4))
        materialized.position.weight.copy_(quantize_rows(materialized.position.weight, 4))
    return materialized


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
