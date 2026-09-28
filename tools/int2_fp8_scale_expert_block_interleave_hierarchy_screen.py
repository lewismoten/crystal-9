"""Screen exact fixed-block corresponding-expert FP8-scale interleave transport."""

from __future__ import annotations

import hashlib
import json
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
BLOCK_SIZE = 32


def encode_expert_block_interleave(streams: tuple[bytes, ...], block_size: int) -> bytes:
    """Emit same-offset fixed blocks from each equal-length expert stream."""
    if block_size < 1 or not streams or not streams[0] or any(len(stream) != len(streams[0]) for stream in streams):
        raise ValueError("expert streams must be non-empty, equal length, with a positive block size")
    return b"".join(
        stream[offset : offset + block_size]
        for offset in range(0, len(streams[0]), block_size)
        for stream in streams
    )


def decode_expert_block_interleave(payload: bytes, expert_count: int, stream_length: int, block_size: int) -> tuple[bytes, ...]:
    """Invert fixed-block corresponding-expert interleave."""
    if expert_count < 1 or stream_length < 1 or block_size < 1 or len(payload) != expert_count * stream_length:
        raise ValueError("interleaved payload length does not match expert stream dimensions")
    streams = [bytearray() for _ in range(expert_count)]
    cursor = 0
    for offset in range(0, stream_length, block_size):
        length = min(block_size, stream_length - offset)
        for stream in streams:
            stream.extend(payload[cursor : cursor + length])
            cursor += length
    return tuple(bytes(stream) for stream in streams)


def _scale_bytes(value: torch.Tensor) -> bytes:
    return value.detach().abs().to(torch.float8_e4m3fn).view(torch.uint8).contiguous().numpy().tobytes()


def screen() -> dict:
    checkpoint_path = Path("artifacts-fp32.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size=13).eval()
    source.load_state_dict(checkpoint["state_dict"])
    state = source.state_dict()
    expert_streams: dict[str, list[bytes]] = {}
    non_expert_parts = []
    for name, value in state.items():
        if name.startswith("experts."):
            _, _, suffix = name.split(".", 2)
            expert_streams.setdefault(suffix, []).append(_scale_bytes(value))
        else:
            non_expert_parts.append(_scale_bytes(value))
    parts = list(non_expert_parts)
    for suffix in sorted(expert_streams):
        streams = tuple(expert_streams[suffix])
        encoded = encode_expert_block_interleave(streams, BLOCK_SIZE)
        if decode_expert_block_interleave(encoded, len(streams), len(streams[0]), BLOCK_SIZE) != streams:
            raise ValueError("expert-block scale reconstruction failed")
        parts.append(encoded)
    scale_bytes = len(lzma.compress(b"".join(parts), preset=9))
    lower_bound = scale_bytes + ACCEPTED_CODE_BYTES + FIXED_OVERHEAD_BYTES
    improves = scale_bytes < ACCEPTED_SCALE_BYTES and lower_bound < ACCEPTED_ARTIFACT_BYTES
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-expert-block-interleave-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": ACCEPTED_ARTIFACT_BYTES, "compressed_scale_bytes": ACCEPTED_SCALE_BYTES, "compressed_code_bytes": ACCEPTED_CODE_BYTES, "container_overhead_bytes": FIXED_OVERHEAD_BYTES},
        "quantization": {"bits": 2, "scale_count": sum(value.numel() for value in state.values()), "scale_layout": "exact FP8 scalar scales from corresponding expert tensors interleaved as fixed 32-byte blocks, with non-expert tensors canonical, then LZMA-9", "block_size": BLOCK_SIZE, "storage_efficient": False},
        "screen": {"expert_block_interleave_lzma_scale_bytes": scale_bytes, "lower_bound_container_bytes": lower_bound, "delta_from_accepted_container_bytes": lower_bound - ACCEPTED_ARTIFACT_BYTES},
        "verification": {"expert_block_interleave_reconstructs_exact_scale_stream": True},
        "decision": "advance-to-runtime" if improves else "rejected-before-runtime",
        "reason": "The exact fixed-block expert interleave improves the accepted complete container and requires a separately parity-tested runtime candidate." if improves else "The fixed-block expert interleave does not improve the accepted complete container; no runtime, artifact, or exhaustive policy evaluation was created.",
    }


if __name__ == "__main__":
    report = screen()
    output = Path("artifacts/rejected/int2-packed-scalar-fp8-scale-expert-block-interleave-lzma-hierarchy-screen-20260928/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], sort_keys=True))
