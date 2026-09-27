"""Exact row-local modular FP8-scale hierarchy screen."""

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


def encode_row_modular_residuals(rows: tuple[bytes, ...]) -> tuple[bytes, bytes]:
    """Store each row root and exact bytewise modular successor residuals."""
    roots = bytearray()
    residuals = bytearray()
    for row in rows:
        if not row:
            raise ValueError("scale rows must not be empty")
        roots.append(row[0])
        residuals.extend((current - previous) % 256 for previous, current in zip(row, row[1:]))
    return bytes(roots), bytes(residuals)


def decode_row_modular_residuals(row_lengths, roots: bytes, residuals: bytes) -> tuple[bytes, ...]:
    """Invert row-local modular residuals without changing any FP8 byte."""
    lengths = tuple(row_lengths)
    if any(length < 1 for length in lengths):
        raise ValueError("scale rows must not be empty")
    if len(roots) != len(lengths):
        raise ValueError("root inventory does not match scale rows")
    if len(residuals) != sum(length - 1 for length in lengths):
        raise ValueError("residual length does not match scale rows")
    offset = 0
    rows = []
    for length, root in zip(lengths, roots):
        row = bytearray([root])
        for residual in residuals[offset : offset + length - 1]:
            row.append((row[-1] + residual) % 256)
        rows.append(bytes(row))
        offset += length - 1
    return tuple(rows)


def _scale_rows(source: TinyMoEPolicy) -> tuple[bytes, ...]:
    rows = []
    for value in source.state_dict().values():
        scales = value.detach().abs().to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        if scales.ndim == 0:
            rows.append(scales.numpy().tobytes())
        else:
            rows.extend(row.contiguous().numpy().tobytes() for row in scales.reshape(scales.shape[0], -1))
    return tuple(rows)


def screen() -> dict:
    checkpoint_path = Path("artifacts-fp32.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size=13).eval()
    source.load_state_dict(checkpoint["state_dict"])
    rows = _scale_rows(source)
    canonical = b"".join(rows)
    roots, residuals = encode_row_modular_residuals(rows)
    if decode_row_modular_residuals((len(row) for row in rows), roots, residuals) != rows:
        raise ValueError("row-modular scale reconstruction failed")
    canonical_bytes = len(lzma.compress(canonical, preset=9))
    hierarchy_bytes = len(lzma.compress(roots + residuals, preset=9))
    lower_bound = hierarchy_bytes + ACCEPTED_CODE_BYTES + FIXED_OVERHEAD_BYTES
    improves = hierarchy_bytes < ACCEPTED_SCALE_BYTES and lower_bound < ACCEPTED_ARTIFACT_BYTES
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-row-modular-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": ACCEPTED_ARTIFACT_BYTES, "compressed_scale_bytes": ACCEPTED_SCALE_BYTES, "compressed_code_bytes": ACCEPTED_CODE_BYTES, "container_overhead_bytes": FIXED_OVERHEAD_BYTES},
        "quantization": {"bits": 2, "scale_count": len(canonical), "scale_layout": "one exact FP8 root per parameter row plus bytewise modular successor residuals, then LZMA-9", "storage_efficient": False},
        "screen": {"row_count": len(rows), "root_bytes": len(roots), "residual_bytes": len(residuals), "canonical_lzma_scale_bytes": canonical_bytes, "row_modular_lzma_scale_bytes": hierarchy_bytes, "lower_bound_container_bytes": lower_bound, "delta_from_accepted_container_bytes": lower_bound - ACCEPTED_ARTIFACT_BYTES},
        "verification": {"row_modular_residual_stream_reconstructs_exact_scale_stream": True, "source_scale_count_matches_fixed_runtime": len(canonical) == 24726},
        "decision": "advance-to-runtime" if improves else "rejected-before-runtime",
        "reason": "The exact row-modular hierarchy improves the accepted complete container and requires a separately parity-tested runtime candidate." if improves else "The exact row-modular hierarchy does not improve the accepted complete container; no runtime, artifact, or exhaustive policy evaluation was created.",
    }


if __name__ == "__main__":
    report = screen()
    output = Path("artifacts/rejected/int2-packed-scalar-fp8-scale-row-modular-lzma-hierarchy-screen-20260927/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], sort_keys=True))
