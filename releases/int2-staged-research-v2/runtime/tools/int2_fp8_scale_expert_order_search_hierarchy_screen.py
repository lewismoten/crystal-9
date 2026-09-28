"""Screen deterministic corresponding-expert byte orders for exact FP8 scales."""

from __future__ import annotations

import hashlib
import lzma
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal9 import TinyMoEPolicy


ACCEPTED_ARTIFACT_BYTES = 20741
ACCEPTED_SCALE_BYTES = 17524
ACCEPTED_CODE_BYTES = 3178
FIXED_OVERHEAD_BYTES = 39


def _validate_order(expert_count: int, order: tuple[int, ...]) -> None:
    if expert_count < 1 or tuple(sorted(order)) != tuple(range(expert_count)):
        raise ValueError("order must contain each expert exactly once")


def encode_ordered_expert_interleave(streams: tuple[bytes, ...], order: tuple[int, ...]) -> bytes:
    """Interleave equal-length expert streams in a reversible supplied order."""
    if not streams or not streams[0] or any(len(stream) != len(streams[0]) for stream in streams):
        raise ValueError("expert streams must be non-empty and equal length")
    _validate_order(len(streams), order)
    return b"".join(streams[index][offset : offset + 1] for offset in range(len(streams[0])) for index in order)


def decode_ordered_expert_interleave(payload: bytes, expert_count: int, stream_length: int, order: tuple[int, ...]) -> tuple[bytes, ...]:
    """Invert a supplied corresponding-expert byte order."""
    _validate_order(expert_count, order)
    if stream_length < 1 or len(payload) != expert_count * stream_length:
        raise ValueError("interleaved payload length does not match expert stream dimensions")
    streams = [bytearray(stream_length) for _ in range(expert_count)]
    cursor = 0
    for offset in range(stream_length):
        for index in order:
            streams[index][offset] = payload[cursor]
            cursor += 1
    return tuple(bytes(stream) for stream in streams)


def _scale_bytes(value: torch.Tensor) -> bytes:
    return value.detach().abs().to(torch.float8_e4m3fn).view(torch.uint8).contiguous().numpy().tobytes()


def screen(orders: tuple[tuple[int, ...], ...]) -> dict:
    """Measure supplied reversible expert orders against the accepted transport."""
    checkpoint_path = Path("artifacts-fp32.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size=13).eval()
    source.load_state_dict(checkpoint["state_dict"])
    state = source.state_dict()
    expert_streams: dict[str, list[bytes]] = {}
    non_expert_parts: list[bytes] = []
    for name, value in state.items():
        if name.startswith("experts."):
            _, _, suffix = name.split(".", 2)
            expert_streams.setdefault(suffix, []).append(_scale_bytes(value))
        else:
            non_expert_parts.append(_scale_bytes(value))
    candidates = []
    for order in orders:
        parts = list(non_expert_parts)
        for suffix in sorted(expert_streams):
            streams = tuple(expert_streams[suffix])
            encoded = encode_ordered_expert_interleave(streams, order)
            if decode_ordered_expert_interleave(encoded, len(streams), len(streams[0]), order) != streams:
                raise ValueError("ordered expert scale reconstruction failed")
            parts.append(encoded)
        candidates.append((len(lzma.compress(b"".join(parts), preset=9)), order))
    best_scale_bytes, best_order = min(candidates)
    lower_bound = best_scale_bytes + ACCEPTED_CODE_BYTES + FIXED_OVERHEAD_BYTES
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-expert-order-search-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": ACCEPTED_ARTIFACT_BYTES, "compressed_scale_bytes": ACCEPTED_SCALE_BYTES, "compressed_code_bytes": ACCEPTED_CODE_BYTES, "container_overhead_bytes": FIXED_OVERHEAD_BYTES},
        "quantization": {"bits": 2, "scale_count": sum(value.numel() for value in state.values()), "scale_layout": "exact FP8 scalar scales from corresponding expert tensors byte-interleaved in a supplied fixed expert order, with non-expert tensors canonical, then LZMA-9", "storage_efficient": False},
        "screen": {"orders_screened": len(orders), "best_order": list(best_order), "best_lzma_scale_bytes": best_scale_bytes, "lower_bound_container_bytes": lower_bound, "delta_from_accepted_container_bytes": lower_bound - ACCEPTED_ARTIFACT_BYTES},
        "verification": {"all_candidate_orders_reconstruct_exact_scale_stream": True},
        "decision": "advance-to-runtime" if lower_bound < ACCEPTED_ARTIFACT_BYTES else "rejected-before-runtime",
    }
