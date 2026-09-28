"""Screen a fixed far-expert ordering for exact FP8-scale interleave transport."""

from __future__ import annotations

import hashlib
import json
import lzma
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal9 import TinyMoEPolicy
from tools.int2_fp8_scale_expert_interleave_hierarchy_screen import encode_expert_interleave

ACCEPTED_ARTIFACT_BYTES = 20741
ACCEPTED_SCALE_BYTES = 17524
ACCEPTED_CODE_BYTES = 3178
FIXED_OVERHEAD_BYTES = 39
EXPERT_ORDER = (0, 8, 1, 7, 2, 6, 3, 5, 4)


def _validate_order(order: tuple[int, ...], count: int) -> None:
    if tuple(sorted(order)) != tuple(range(count)):
        raise ValueError("expert order must be a complete permutation")


def permute_expert_streams(streams: tuple[bytes, ...], order: tuple[int, ...]) -> tuple[bytes, ...]:
    """Reorder corresponding expert streams by an explicit reversible permutation."""
    _validate_order(order, len(streams))
    return tuple(streams[index] for index in order)


def restore_expert_streams(ordered: tuple[bytes, ...], order: tuple[int, ...]) -> tuple[bytes, ...]:
    """Restore canonical expert order from an explicit reversible permutation."""
    _validate_order(order, len(ordered))
    restored = [b""] * len(ordered)
    for destination, source in enumerate(order):
        restored[source] = ordered[destination]
    return tuple(restored)


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
        ordered = permute_expert_streams(streams, EXPERT_ORDER)
        if restore_expert_streams(ordered, EXPERT_ORDER) != streams:
            raise ValueError("expert-order scale reconstruction failed")
        parts.append(encode_expert_interleave(ordered))
    scale_bytes = len(lzma.compress(b"".join(parts), preset=9))
    lower_bound = scale_bytes + ACCEPTED_CODE_BYTES + FIXED_OVERHEAD_BYTES
    improves = scale_bytes < ACCEPTED_SCALE_BYTES and lower_bound < ACCEPTED_ARTIFACT_BYTES
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-far-expert-order-interleave-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": ACCEPTED_ARTIFACT_BYTES, "compressed_scale_bytes": ACCEPTED_SCALE_BYTES, "compressed_code_bytes": ACCEPTED_CODE_BYTES, "container_overhead_bytes": FIXED_OVERHEAD_BYTES},
        "quantization": {"bits": 2, "scale_count": sum(value.numel() for value in state.values()), "scale_layout": "exact corresponding-expert FP8 scalar scales byte-interleaved after fixed far-expert ordering", "expert_order": list(EXPERT_ORDER), "storage_efficient": False},
        "screen": {"far_expert_order_lzma_scale_bytes": scale_bytes, "lower_bound_container_bytes": lower_bound, "delta_from_accepted_container_bytes": lower_bound - ACCEPTED_ARTIFACT_BYTES},
        "verification": {"expert_order_reconstructs_exact_scale_stream": True},
        "decision": "advance-to-runtime" if improves else "rejected-before-runtime",
        "reason": "The fixed far-expert ordering improves the complete container and requires a separately parity-tested runtime candidate." if improves else "The fixed far-expert ordering does not improve the accepted complete container; no runtime, artifact, or exhaustive policy evaluation was created.",
    }


if __name__ == "__main__":
    report = screen()
    output = Path("artifacts/rejected/int2-packed-scalar-fp8-scale-far-expert-order-interleave-lzma-hierarchy-screen-20260928/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], sort_keys=True))
