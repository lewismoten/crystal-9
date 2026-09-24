"""Crystal-9 tokenizer and model primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed [-8, 7] values low-nibble first; pad an odd final value."""
    if values.dtype != torch.int8 or values.ndim != 1:
        raise ValueError("INT4 packing requires a rank-1 torch.int8 tensor")
    if torch.any(values < -8) or torch.any(values > 7):
        raise ValueError("INT4 values must be in [-8, 7]")
    unsigned = (values.to(torch.int16) & 0x0F).to(torch.uint8)
    if unsigned.numel() % 2:
        unsigned = torch.cat((unsigned, torch.zeros(1, dtype=torch.uint8, device=values.device)))
    return unsigned[0::2] | (unsigned[1::2] << 4)


def unpack_signed_int4(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack low-nibble-first signed INT4 values, discarding final padding."""
    if packed.dtype != torch.uint8 or packed.ndim != 1 or count < 0 or count > packed.numel() * 2:
        raise ValueError("invalid packed INT4 buffer or count")
    unsigned = torch.empty(packed.numel() * 2, dtype=torch.int16, device=packed.device)
    unsigned[0::2] = packed.to(torch.int16) & 0x0F
    unsigned[1::2] = packed.to(torch.int16) >> 4
    return torch.where(unsigned[:count] >= 8, unsigned[:count] - 16, unsigned[:count]).to(torch.int8)


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


def quantize_row_groups(values: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Quantize each contiguous group within every matrix row independently."""
    if values.ndim != 2:
        raise ValueError("row-group quantization requires a rank-2 tensor")
    if group_size < 1 or values.shape[1] % group_size:
        raise ValueError("group size must divide each row width")
    return quantize_rows(values.reshape(-1, group_size), bits).reshape_as(values)


def quantize_row_groups_ste(values: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    quantized = quantize_row_groups(values, bits, group_size)
    return values + (quantized - values).detach()


def quantize_groups(values: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Symmetrically quantize a vector using a separate scale per group."""
    if values.ndim != 1:
        raise ValueError("group quantization requires a rank-1 tensor")
    if group_size < 1 or values.numel() % group_size:
        raise ValueError("group size must divide the vector length")
    return quantize_rows(values.reshape(-1, group_size), bits).reshape_as(values)


def quantize_groups_ste(values: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    quantized = quantize_groups(values, bits, group_size)
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

    def forward_mixed_int3_suffix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """First INT3 tracer: rowwise INT3 routed-expert and output matrices only."""
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
            value = F.linear(state, quantize_rows_ste(first.weight, 3), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 3), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), self.output.bias)

    def forward_mixed_int3_suffix_output_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Second INT3 tracer: accepted suffix plus INT3 final-output bias."""
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
            value = F.linear(state, quantize_rows_ste(first.weight, 3), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 3), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), quantize_ste(self.output.bias, 3))

    def forward_mixed_int3_suffix_output_bias_router_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Scoped INT3 suffix with the router bias as the next isolated group."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position(positions)
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(F.linear(state, self.router.weight, quantize_ste(self.router.bias, 3)), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 3), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 3), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), quantize_ste(self.output.bias, 3))

    def forward_mixed_int3_suffix_output_bias_router_weight(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Scoped INT3 suffix with rowwise INT3 router weight."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position(positions)
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(F.linear(state, quantize_rows_ste(self.router.weight, 3), self.router.bias), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 3), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 3), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), quantize_ste(self.output.bias, 3))

    def forward_mixed_int3_suffix_output_bias_router_weight_group4(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Scoped INT3 suffix with four-value groupwise INT3 router weights."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position(positions)
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(F.linear(state, quantize_row_groups_ste(self.router.weight, 3, 4), self.router.bias), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 3), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 3), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), quantize_ste(self.output.bias, 3))

    def forward_mixed_int3_suffix_output_bias_router_weight_group4_router_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Scoped INT3 suffix with groupwise router weight and INT3 router bias."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position(positions)
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(F.linear(state, quantize_row_groups_ste(self.router.weight, 3, 4), quantize_ste(self.router.bias, 3)), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 3), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 3), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), quantize_ste(self.output.bias, 3))

    def forward_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Scoped INT3 suffix with groupwise router weight, router bias, and expert biases."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position(positions)
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(F.linear(state, quantize_row_groups_ste(self.router.weight, 3, 4), quantize_ste(self.router.bias, 3)), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 3), quantize_ste(first.bias, 3))
            expert_outputs.append(F.linear(F.silu(value), quantize_rows_ste(second.weight, 3), quantize_ste(second.bias, 3)))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), quantize_ste(self.output.bias, 3))

    def forward_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_input_group4(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Accepted INT3 scope plus four-value groupwise INT3 input tables."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = F.embedding(token_ids, quantize_row_groups_ste(self.embedding.weight, 3, 4), padding_idx=0)
        hidden = hidden + F.embedding(positions, quantize_row_groups_ste(self.position.weight, 3, 4))
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(F.linear(state, quantize_row_groups_ste(self.router.weight, 3, 4), quantize_ste(self.router.bias, 3)), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 3), quantize_ste(first.bias, 3))
            expert_outputs.append(F.linear(F.silu(value), quantize_rows_ste(second.weight, 3), quantize_ste(second.bias, 3)))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), quantize_ste(self.output.bias, 3))

    def forward_mixed_int3_suffix_input(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Second INT3 tracer: INT3 input tables plus the accepted INT3 suffix."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = F.embedding(token_ids, quantize_rows_ste(self.embedding.weight, 3), padding_idx=0)
        hidden = hidden + F.embedding(positions, quantize_rows_ste(self.position.weight, 3))
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, key_padding_mask=token_ids.eq(0), need_weights=False)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        state = self.norm(attended[torch.arange(token_ids.shape[0], device=token_ids.device), last])
        router_weights = torch.softmax(self.router(state), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 3), first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 3), second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 3), self.output.bias)

    def forward_mixed_int4_with_expert_biases(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Baseline mixed INT4 layout with every routed-expert bias at INT4."""
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
            value = F.linear(state, quantize_rows_ste(first.weight, 4), quantize_ste(first.bias, 4))
            expert_outputs.append(F.linear(F.silu(value), quantize_rows_ste(second.weight, 4), quantize_ste(second.bias, 4)))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        return F.linear(state + (routed * top_weights.unsqueeze(-1)).sum(dim=1), quantize_rows_ste(self.output.weight, 4), self.output.bias)

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

    def _int4_self_attention(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        int4_groups: frozenset[str] = frozenset({"q", "k", "v", "out"}),
        quantize_attention_biases: bool = False,
        attention_bias_groups: frozenset[str] | None = None,
    ) -> torch.Tensor:
        """Custom runtime path with an explicit INT4-row attention inventory."""
        attention_bias_groups = attention_bias_groups or (frozenset({"in", "out"}) if quantize_attention_biases else frozenset())
        batch, steps, width = hidden.shape
        heads = self.attention.num_heads
        head_width = width // heads
        projection_weight = self.attention.in_proj_weight
        if int4_groups & {"q", "k", "v"}:
            projection_weight = projection_weight.clone()
            for index, group in enumerate(("q", "k", "v")):
                if group in int4_groups:
                    rows = slice(index * width, (index + 1) * width)
                    projection_weight[rows] = quantize_rows_ste(projection_weight[rows], 4)
        input_bias = quantize_ste(self.attention.in_proj_bias, 4) if "in" in attention_bias_groups else self.attention.in_proj_bias
        qkv = F.linear(hidden, projection_weight, input_bias)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, steps, heads, head_width).transpose(1, 2)
        key = key.view(batch, steps, heads, head_width).transpose(1, 2)
        value = value.view(batch, steps, heads, head_width).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) * (head_width ** -0.5)
        causal = torch.triu(torch.ones(steps, steps, device=hidden.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        scores = scores.masked_fill(token_ids.eq(0).view(batch, 1, 1, steps), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, steps, width)
        output_weight = quantize_rows_ste(self.attention.out_proj.weight, 4) if "out" in int4_groups else self.attention.out_proj.weight
        output_bias = quantize_ste(self.attention.out_proj.bias, 4) if "out" in attention_bias_groups else self.attention.out_proj.bias
        return F.linear(attended, output_weight, output_bias)

    def forward_mixed_int4_input_attention_groups(
        self,
        token_ids: torch.Tensor,
        int4_groups: frozenset[str],
        quantize_attention_biases: bool = False,
        attention_bias_groups: frozenset[str] | None = None,
        quantize_router_weight: bool = False,
        quantize_router_bias: bool = False,
        quantize_expert_biases: bool = False,
        quantize_output_bias: bool = False,
        quantize_norm: bool = False,
        norm_int4_groups: frozenset[str] | None = None,
        norm_weight_group_size: int | None = None,
    ) -> torch.Tensor:
        """INT4-row input tables plus explicitly selected attention projection groups."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        hidden = F.embedding(token_ids, quantize_rows_ste(self.embedding.weight, 4), padding_idx=0)
        hidden = hidden + F.embedding(positions, quantize_rows_ste(self.position.weight, 4))
        attended = self._int4_self_attention(hidden, token_ids, int4_groups, quantize_attention_biases, attention_bias_groups)
        last = (token_ids.ne(0).sum(dim=1) - 1).clamp(min=0)
        selected = attended[torch.arange(token_ids.shape[0], device=token_ids.device), last]
        norm_int4_groups = norm_int4_groups if norm_int4_groups is not None else (frozenset({"weight", "bias"}) if quantize_norm else frozenset())
        norm_weight = self.norm.weight
        if "weight" in norm_int4_groups:
            norm_weight = quantize_groups_ste(norm_weight, 4, norm_weight_group_size) if norm_weight_group_size else quantize_ste(norm_weight, 4)
        norm_bias = quantize_ste(self.norm.bias, 4) if "bias" in norm_int4_groups else self.norm.bias
        state = F.layer_norm(selected, self.norm.normalized_shape, norm_weight, norm_bias, self.norm.eps)
        router_weight = quantize_rows_ste(self.router.weight, 4) if quantize_router_weight else self.router.weight
        router_bias = quantize_ste(self.router.bias, 4) if quantize_router_bias else self.router.bias
        router_weights = torch.softmax(F.linear(state, router_weight, router_bias), dim=-1)
        top_weights, top_indices = router_weights.topk(2, dim=-1)
        expert_outputs = []
        for expert in self.experts:
            first, _, second = expert
            value = F.linear(state, quantize_rows_ste(first.weight, 4), quantize_ste(first.bias, 4) if quantize_expert_biases else first.bias)
            value = F.silu(value)
            expert_outputs.append(F.linear(value, quantize_rows_ste(second.weight, 4), quantize_ste(second.bias, 4) if quantize_expert_biases else second.bias))
        all_experts = torch.stack(expert_outputs, dim=1)
        routed = all_experts.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
        state = state + (routed * top_weights.unsqueeze(-1)).sum(dim=1)
        return F.linear(state, quantize_rows_ste(self.output.weight, 4), quantize_ste(self.output.bias, 4) if quantize_output_bias else self.output.bias)

    def forward_mixed_int4_input_attention_groups_with_biases(
        self,
        token_ids: torch.Tensor,
        int4_groups: frozenset[str],
        quantize_attention_biases: bool = True,
        attention_bias_groups: frozenset[str] | None = None,
    ) -> torch.Tensor:
        """Apply the selected INT4 layout, including attention biases when requested."""
        return self.forward_mixed_int4_input_attention_groups(token_ids, int4_groups, quantize_attention_biases, attention_bias_groups)

    def forward_mixed_int4_input_attention_groups_with_router(
        self,
        token_ids: torch.Tensor,
        int4_groups: frozenset[str],
        attention_bias_groups: frozenset[str] | None = None,
        quantize_router_weight: bool = True,
    ) -> torch.Tensor:
        """Apply the selected INT4 layout with an INT4-row router weight."""
        return self.forward_mixed_int4_input_attention_groups(
            token_ids, int4_groups, attention_bias_groups is not None, attention_bias_groups, quantize_router_weight
        )

    def forward_mixed_int4_input_attention_q(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_mixed_int4_input_attention_groups(token_ids, frozenset({"q"}))

    def forward_mixed_int4_input_attention(self, token_ids: torch.Tensor) -> torch.Tensor:
        """INT4-row input tables and all attention projections with a fixed INT4 suffix."""
        return self.forward_mixed_int4_input_attention_groups(token_ids, frozenset({"q", "k", "v", "out"}))

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


def materialize_mixed_int3_suffix(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the first scoped INT3 layout exactly."""
    materialized = TinyMoEPolicy(
        source.embedding.num_embeddings,
        source.embedding.embedding_dim,
        len(source.experts),
    ).to(next(source.parameters()).device)
    materialized.load_state_dict(source.state_dict())
    with torch.no_grad():
        for expert in materialized.experts:
            expert[0].weight.copy_(quantize_rows(expert[0].weight, 3))
            expert[2].weight.copy_(quantize_rows(expert[2].weight, 3))
        materialized.output.weight.copy_(quantize_rows(materialized.output.weight, 3))
    return materialized


def materialize_mixed_int3_suffix_output_bias(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the scoped INT3 suffix plus final-output bias."""
    materialized = materialize_mixed_int3_suffix(source)
    with torch.no_grad():
        materialized.output.bias.copy_(quantize_tensor(materialized.output.bias, 3))
    return materialized


def materialize_mixed_int3_suffix_output_bias_router_bias(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the scoped INT3 suffix, output bias, and router bias."""
    materialized = materialize_mixed_int3_suffix_output_bias(source)
    with torch.no_grad():
        materialized.router.bias.copy_(quantize_tensor(materialized.router.bias, 3))
    return materialized


def materialize_mixed_int3_suffix_output_bias_router_weight(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the scoped INT3 suffix, output bias, and router weight."""
    materialized = materialize_mixed_int3_suffix_output_bias(source)
    with torch.no_grad():
        materialized.router.weight.copy_(quantize_rows(materialized.router.weight, 3))
    return materialized


def materialize_mixed_int3_suffix_output_bias_router_weight_group4(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the scoped INT3 suffix plus four-value router-weight groups."""
    materialized = materialize_mixed_int3_suffix_output_bias(source)
    with torch.no_grad():
        materialized.router.weight.copy_(quantize_row_groups(materialized.router.weight, 3, 4))
    return materialized


def materialize_mixed_int3_suffix_output_bias_router_weight_group4_router_bias(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize groupwise INT3 router weights plus the INT3 router bias."""
    materialized = materialize_mixed_int3_suffix_output_bias_router_weight_group4(source)
    with torch.no_grad():
        materialized.router.bias.copy_(quantize_tensor(materialized.router.bias, 3))
    return materialized


def materialize_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the groupwise-router stage with every routed-expert bias at INT3."""
    materialized = materialize_mixed_int3_suffix_output_bias_router_weight_group4_router_bias(source)
    with torch.no_grad():
        for expert in materialized.experts:
            expert[0].bias.copy_(quantize_tensor(expert[0].bias, 3))
            expert[2].bias.copy_(quantize_tensor(expert[2].bias, 3))
    return materialized


def materialize_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_input_group4(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted INT3 scope with four-value groupwise input tables."""
    materialized = materialize_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases(source)
    with torch.no_grad():
        materialized.embedding.weight.copy_(quantize_row_groups(materialized.embedding.weight, 3, 4))
        materialized.position.weight.copy_(quantize_row_groups(materialized.position.weight, 3, 4))
    return materialized


def materialize_mixed_int3_suffix_input(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the scoped INT3 suffix plus rowwise INT3 input tables."""
    materialized = materialize_mixed_int3_suffix(source)
    with torch.no_grad():
        materialized.embedding.weight.copy_(quantize_rows(materialized.embedding.weight, 3))
        materialized.position.weight.copy_(quantize_rows(materialized.position.weight, 3))
    return materialized


def materialize_mixed_int4_with_expert_biases(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the mixed INT4 layout including every routed-expert bias."""
    materialized = materialize_mixed_int4(source)
    with torch.no_grad():
        for expert in materialized.experts:
            expert[0].bias.copy_(quantize_tensor(expert[0].bias, 4))
            expert[2].bias.copy_(quantize_tensor(expert[2].bias, 4))
    return materialized


def materialize_mixed_int4_input(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the input-table stage plus the frozen mixed-INT4 suffix."""
    materialized = materialize_mixed_int4(source)
    with torch.no_grad():
        materialized.embedding.weight.copy_(quantize_rows(materialized.embedding.weight, 4))
        materialized.position.weight.copy_(quantize_rows(materialized.position.weight, 4))
    return materialized


def materialize_mixed_int4_input_attention_groups(source: TinyMoEPolicy, int4_groups: frozenset[str]) -> TinyMoEPolicy:
    """Materialize input INT4 plus the specified attention projections."""
    materialized = materialize_mixed_int4_input(source)
    width = materialized.attention.embed_dim
    with torch.no_grad():
        for index, group in enumerate(("q", "k", "v")):
            if group in int4_groups:
                rows = slice(index * width, (index + 1) * width)
                materialized.attention.in_proj_weight[rows].copy_(quantize_rows(materialized.attention.in_proj_weight[rows], 4))
        if "out" in int4_groups:
            materialized.attention.out_proj.weight.copy_(quantize_rows(materialized.attention.out_proj.weight, 4))
    return materialized


def materialize_mixed_int4_input_attention_q(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the input stage plus only the INT4-row Q projection."""
    return materialize_mixed_int4_input_attention_groups(source, frozenset({"q"}))


def materialize_mixed_int4_input_attention_v(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted Q stage plus the INT4-row V projection."""
    return materialize_mixed_int4_input_attention_groups(source, frozenset({"q", "v"}))


def materialize_mixed_int4_input_attention_q_v_out(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the Q/V stage plus the INT4-row attention output projection."""
    return materialize_mixed_int4_input_attention_groups(source, frozenset({"q", "v", "out"}))


def materialize_mixed_int4_input_attention_q_v_out_k(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize Q/V/output plus the final INT4-row K projection."""
    return materialize_mixed_int4_input_attention_groups(source, frozenset({"q", "k", "v", "out"}))


def materialize_mixed_int4_input_attention_q_v_out_k_output_bias(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted all-attention weight layout plus only its output bias."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k(source)
    with torch.no_grad():
        materialized.attention.out_proj.bias.copy_(quantize_tensor(materialized.attention.out_proj.bias, 4))
    return materialized


def materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted attention layout plus INT4-row router weight."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k_output_bias(source)
    with torch.no_grad():
        materialized.router.weight.copy_(quantize_rows(materialized.router.weight, 4))
    return materialized


def materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted router layout plus the attention input bias."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight(source)
    with torch.no_grad():
        materialized.attention.in_proj_bias.copy_(quantize_tensor(materialized.attention.in_proj_bias, 4))
    return materialized


def materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted input-bias layout plus an INT4 router bias."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias(source)
    with torch.no_grad():
        materialized.router.bias.copy_(quantize_tensor(materialized.router.bias, 4))
    return materialized


def materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias_expert_biases(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted router-bias layout plus every routed-expert bias."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias(source)
    with torch.no_grad():
        for expert in materialized.experts:
            expert[0].bias.copy_(quantize_tensor(expert[0].bias, 4))
            expert[2].bias.copy_(quantize_tensor(expert[2].bias, 4))
    return materialized


def materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias_expert_biases_output_bias(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted expert-bias layout plus output bias."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias_expert_biases(source)
    with torch.no_grad():
        materialized.output.bias.copy_(quantize_tensor(materialized.output.bias, 4))
    return materialized


def materialize_mixed_int4_norm_bias(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the accepted output-bias layout plus only LayerNorm bias."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias_expert_biases_output_bias(source)
    with torch.no_grad():
        materialized.norm.bias.copy_(quantize_tensor(materialized.norm.bias, 4))
    return materialized


def materialize_mixed_int4_full_parameters_grouped_norm_weight(source: TinyMoEPolicy, group_size: int = 8) -> TinyMoEPolicy:
    """Materialize all parameters with LayerNorm weight INT4 in fixed-size groups."""
    materialized = materialize_mixed_int4_norm_bias(source)
    with torch.no_grad():
        materialized.norm.weight.copy_(quantize_groups(materialized.norm.weight, 4, group_size))
    return materialized


def materialize_mixed_int4_full_parameters(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize every persisted Crystal-9 parameter at its exact INT4 layout."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k_output_bias_router_weight_input_bias_router_bias_expert_biases_output_bias(source)
    with torch.no_grad():
        materialized.norm.weight.copy_(quantize_tensor(materialized.norm.weight, 4))
        materialized.norm.bias.copy_(quantize_tensor(materialized.norm.bias, 4))
    return materialized


def materialize_mixed_int4_input_attention_q_v_out_k_attention_biases(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize every attention weight and both attention bias tensors as INT4."""
    materialized = materialize_mixed_int4_input_attention_q_v_out_k(source)
    with torch.no_grad():
        materialized.attention.in_proj_bias.copy_(quantize_tensor(materialized.attention.in_proj_bias, 4))
        materialized.attention.out_proj.bias.copy_(quantize_tensor(materialized.attention.out_proj.bias, 4))
    return materialized


def materialize_mixed_int4_input_attention(source: TinyMoEPolicy) -> TinyMoEPolicy:
    """Materialize the input and attention INT4-row stage."""
    materialized = materialize_mixed_int4_input(source)
    with torch.no_grad():
        materialized.attention.in_proj_weight.copy_(quantize_rows(materialized.attention.in_proj_weight, 4))
        materialized.attention.out_proj.weight.copy_(quantize_rows(materialized.attention.out_proj.weight, 4))
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
