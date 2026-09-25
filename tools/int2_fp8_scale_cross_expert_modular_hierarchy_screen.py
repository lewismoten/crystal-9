"""Exact modular cross-expert FP8-scale hierarchy screen."""

from __future__ import annotations

import hashlib
import json
import lzma
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal9 import TinyMoEPolicy


def encode_modular_residual(root: bytes, peer: bytes) -> bytes:
    if len(root) != len(peer):
        raise ValueError("corresponding expert scale streams must have equal length")
    return bytes((value - base) % 256 for base, value in zip(root, peer))


def decode_modular_residual(root: bytes, residual: bytes) -> bytes:
    if len(root) != len(residual):
        raise ValueError("corresponding expert residual streams must have equal length")
    return bytes((base + delta) % 256 for base, delta in zip(root, residual))


def transform_cross_expert_streams(streams: dict[str, bytes]) -> tuple[dict[str, bytes], tuple[str, ...]]:
    transformed = dict(streams)
    residual_names = []
    for layer in ("0", "2"):
        for suffix in ("weight", "bias"):
            root_name = f"experts.0.{layer}.{suffix}"
            if root_name not in streams:
                continue
            root = streams[root_name]
            for expert in range(1, 9):
                name = f"experts.{expert}.{layer}.{suffix}"
                if name not in streams:
                    continue
                transformed[name] = encode_modular_residual(root, streams[name])
                if decode_modular_residual(root, transformed[name]) != streams[name]:
                    raise ValueError("cross-expert modular scale reconstruction failed")
                residual_names.append(name)
    return transformed, tuple(residual_names)


def screen() -> dict:
    checkpoint_path = Path("artifacts-fp32.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size=13).eval()
    source.load_state_dict(checkpoint["state_dict"])
    streams = {
        name: value.detach().abs().to(torch.float8_e4m3fn).view(torch.uint8).contiguous().numpy().tobytes()
        for name, value in source.state_dict().items()
    }
    transformed, residual_names = transform_cross_expert_streams(streams)
    canonical = b"".join(streams.values())
    residual = b"".join(transformed.values())
    canonical_bytes = len(lzma.compress(canonical, preset=9))
    candidate_bytes = len(lzma.compress(residual, preset=9))
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-cross-expert-modular-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": 20769, "compressed_scale_bytes": 17552, "compressed_code_bytes": 3178, "container_overhead_bytes": 39},
        "quantization": {"bits": 2, "scale_count": len(canonical), "scale_layout": "lossless modular residuals of corresponding expert tensor scales, then LZMA-9", "storage_efficient": False},
        "screen": {"tensor_streams": len(streams), "cross_expert_residual_streams": len(residual_names), "canonical_lzma_scale_bytes": canonical_bytes, "cross_expert_modular_lzma_scale_bytes": candidate_bytes, "lower_bound_container_bytes": candidate_bytes + 3178 + 39, "delta_from_accepted_container_bytes": candidate_bytes + 3178 + 39 - 20769},
        "verification": {"cross_expert_modular_stream_reconstructs_exact_scale_stream": True, "source_scale_count_matches_fixed_runtime": len(canonical) == 24726},
        "decision": "rejected-before-runtime",
        "reason": "Corresponding expert tensors are not sufficiently correlated under exact modular residual coding; the candidate exceeds the accepted complete container before decoder-format metadata. No runtime, artifact, or exhaustive policy evaluation was created.",
    }


if __name__ == "__main__":
    report = screen()
    output = Path("artifacts/rejected/int2-packed-scalar-fp8-scale-cross-expert-modular-lzma-hierarchy-screen-20260925/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], sort_keys=True))
