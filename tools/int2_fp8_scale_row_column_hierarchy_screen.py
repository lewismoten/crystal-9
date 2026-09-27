"""Exact tensor-column-major FP8-scale hierarchy screen."""

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


def encode_row_column_stream(rows: tuple[bytes, ...]) -> bytes:
    """Transpose a rectangular row stream into a reversible column-major stream."""
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("rows must form a non-empty rectangle")
    return bytes(value for column in zip(*rows) for value in column)


def decode_row_column_stream(payload: bytes, row_count: int, column_count: int) -> tuple[bytes, ...]:
    """Invert a column-major scale stream into its exact row sequence."""
    if row_count < 1 or column_count < 1 or len(payload) != row_count * column_count:
        raise ValueError("column stream length does not match matrix dimensions")
    rows = [bytearray(column_count) for _ in range(row_count)]
    for column in range(column_count):
        start = column * row_count
        for row in range(row_count):
            rows[row][column] = payload[start + row]
    return tuple(bytes(row) for row in rows)


def _tensor_rows(value: torch.Tensor) -> tuple[bytes, ...]:
    scale_bytes = value.detach().abs().to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    if scale_bytes.ndim == 0:
        return (scale_bytes.numpy().tobytes(),)
    return tuple(row.contiguous().numpy().tobytes() for row in scale_bytes.reshape(scale_bytes.shape[0], -1))


def screen() -> dict:
    checkpoint_path = Path("artifacts-fp32.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size=13).eval()
    source.load_state_dict(checkpoint["state_dict"])

    canonical_parts = []
    column_parts = []
    tensor_shapes = []
    for value in source.state_dict().values():
        rows = _tensor_rows(value)
        canonical_parts.extend(rows)
        encoded = encode_row_column_stream(rows)
        if decode_row_column_stream(encoded, len(rows), len(rows[0])) != rows:
            raise ValueError("column-major scale reconstruction failed")
        column_parts.append(encoded)
        tensor_shapes.append((len(rows), len(rows[0])))

    canonical = b"".join(canonical_parts)
    column_major = b"".join(column_parts)
    canonical_bytes = len(lzma.compress(canonical, preset=9))
    hierarchy_bytes = len(lzma.compress(column_major, preset=9))
    lower_bound = hierarchy_bytes + ACCEPTED_CODE_BYTES + FIXED_OVERHEAD_BYTES
    improves = hierarchy_bytes < ACCEPTED_SCALE_BYTES and lower_bound < ACCEPTED_ARTIFACT_BYTES
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-tensor-column-major-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": ACCEPTED_ARTIFACT_BYTES, "compressed_scale_bytes": ACCEPTED_SCALE_BYTES, "compressed_code_bytes": ACCEPTED_CODE_BYTES, "container_overhead_bytes": FIXED_OVERHEAD_BYTES},
        "quantization": {"bits": 2, "scale_count": len(canonical), "scale_layout": "exact FP8 scalar scales transposed independently to tensor column-major order, then LZMA-9", "storage_efficient": False},
        "screen": {"tensor_count": len(tensor_shapes), "canonical_lzma_scale_bytes": canonical_bytes, "column_major_lzma_scale_bytes": hierarchy_bytes, "lower_bound_container_bytes": lower_bound, "delta_from_accepted_container_bytes": lower_bound - ACCEPTED_ARTIFACT_BYTES},
        "verification": {"column_major_stream_reconstructs_exact_scale_stream": True, "source_scale_count_matches_fixed_runtime": len(canonical) == 24726},
        "decision": "advance-to-runtime" if improves else "rejected-before-runtime",
        "reason": "The exact tensor-column-major hierarchy improves the accepted complete container and requires a separately parity-tested runtime candidate." if improves else "The exact tensor-column-major hierarchy does not improve the accepted complete container; no runtime, artifact, or exhaustive policy evaluation was created.",
    }


if __name__ == "__main__":
    report = screen()
    output = Path("artifacts/rejected/int2-packed-scalar-fp8-scale-tensor-column-major-lzma-hierarchy-screen-20260927/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], sort_keys=True))
