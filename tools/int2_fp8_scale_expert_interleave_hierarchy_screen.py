"""Exact corresponding-expert-interleave FP8-scale hierarchy screen."""

from __future__ import annotations

import hashlib
import json
import lzma
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal9 import TinyMoEPolicy


ACCEPTED_ARTIFACT_BYTES = 20769
ACCEPTED_SCALE_BYTES = 17552
ACCEPTED_CODE_BYTES = 3178
FIXED_OVERHEAD_BYTES = 39


def encode_expert_interleave(streams: tuple[bytes, ...]) -> bytes:
    """Interleave equal-length corresponding expert streams byte by byte."""
    if not streams or not streams[0] or any(len(stream) != len(streams[0]) for stream in streams):
        raise ValueError("expert streams must be non-empty and equal length")
    return bytes(value for values in zip(*streams) for value in values)


def decode_expert_interleave(payload: bytes, expert_count: int, stream_length: int) -> tuple[bytes, ...]:
    """Invert an interleaved corresponding-expert stream."""
    if expert_count < 1 or stream_length < 1 or len(payload) != expert_count * stream_length:
        raise ValueError("interleaved payload length does not match expert stream dimensions")
    streams = [bytearray(stream_length) for _ in range(expert_count)]
    for index, value in enumerate(payload):
        streams[index % expert_count][index // expert_count] = value
    return tuple(bytes(stream) for stream in streams)


def _scale_bytes(value: torch.Tensor) -> bytes:
    return value.detach().abs().to(torch.float8_e4m3fn).view(torch.uint8).contiguous().numpy().tobytes()


def screen() -> dict:
    checkpoint_path = Path("artifacts-fp32.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size=13).eval()
    source.load_state_dict(checkpoint["state_dict"])
    state = source.state_dict()

    canonical = b"".join(_scale_bytes(value) for value in state.values())
    expert_streams: dict[str, list[bytes]] = {}
    non_expert_parts = []
    for name, value in state.items():
        if name.startswith("experts."):
            _, _, suffix = name.split(".", 2)
            expert_streams.setdefault(suffix, []).append(_scale_bytes(value))
        else:
            non_expert_parts.append(_scale_bytes(value))

    interleaved_parts = []
    for suffix in sorted(expert_streams):
        streams = tuple(expert_streams[suffix])
        encoded = encode_expert_interleave(streams)
        if decode_expert_interleave(encoded, len(streams), len(streams[0])) != streams:
            raise ValueError("corresponding-expert scale reconstruction failed")
        interleaved_parts.append(encoded)
    hierarchy = b"".join(non_expert_parts + interleaved_parts)
    canonical_bytes = len(lzma.compress(canonical, preset=9))
    hierarchy_bytes = len(lzma.compress(hierarchy, preset=9))
    lower_bound = hierarchy_bytes + ACCEPTED_CODE_BYTES + FIXED_OVERHEAD_BYTES
    improves = hierarchy_bytes < ACCEPTED_SCALE_BYTES and lower_bound < ACCEPTED_ARTIFACT_BYTES
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-corresponding-expert-interleave-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": ACCEPTED_ARTIFACT_BYTES, "compressed_scale_bytes": ACCEPTED_SCALE_BYTES, "compressed_code_bytes": ACCEPTED_CODE_BYTES, "container_overhead_bytes": FIXED_OVERHEAD_BYTES},
        "quantization": {"bits": 2, "scale_count": len(canonical), "scale_layout": "exact FP8 scalar scales from corresponding expert tensors interleaved bytewise, with non-expert tensors canonical, then LZMA-9", "storage_efficient": False},
        "screen": {"expert_count": 9, "corresponding_expert_tensor_groups": len(expert_streams), "canonical_lzma_scale_bytes": canonical_bytes, "expert_interleaved_lzma_scale_bytes": hierarchy_bytes, "lower_bound_container_bytes": lower_bound, "delta_from_accepted_container_bytes": lower_bound - ACCEPTED_ARTIFACT_BYTES},
        "verification": {"expert_interleave_reconstructs_exact_scale_stream": True, "source_scale_count_matches_fixed_runtime": len(canonical) == 24726},
        "decision": "advance-to-runtime" if improves else "rejected-before-runtime",
        "reason": "The exact corresponding-expert interleave hierarchy improves the accepted complete container and requires a separately parity-tested runtime candidate." if improves else "The exact corresponding-expert interleave hierarchy does not improve the accepted complete container; no runtime, artifact, or exhaustive policy evaluation was created.",
    }


if __name__ == "__main__":
    report = screen()
    output = Path("artifacts/rejected/int2-packed-scalar-fp8-scale-corresponding-expert-interleave-lzma-hierarchy-screen-20260928/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], sort_keys=True))
