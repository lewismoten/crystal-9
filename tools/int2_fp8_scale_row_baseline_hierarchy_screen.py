"""Exact row-mode FP8-scale hierarchy screen."""

from __future__ import annotations

import hashlib
import json
import lzma
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal9 import TinyMoEPolicy


def encode_row_mode_residuals(rows: tuple[bytes, ...]) -> tuple[bytes, bytes]:
    """Store one modal scale byte per row and XOR residuals for exact recovery."""
    baselines = bytearray()
    residual = bytearray()
    for row in rows:
        if not row:
            raise ValueError("scale rows must not be empty")
        counts = Counter(row)
        baseline = min(value for value, count in counts.items() if count == max(counts.values()))
        baselines.append(baseline)
        residual.extend(value ^ baseline for value in row)
    return bytes(baselines), bytes(residual)


def decode_row_mode_residuals(row_lengths, baselines: bytes, residual: bytes) -> tuple[bytes, ...]:
    lengths = tuple(row_lengths)
    if len(baselines) != len(lengths):
        raise ValueError("baseline inventory does not match scale rows")
    if sum(lengths) != len(residual):
        raise ValueError("residual length does not match scale rows")
    offset = 0
    rows = []
    for length, baseline in zip(lengths, baselines):
        segment = residual[offset : offset + length]
        rows.append(bytes(value ^ baseline for value in segment))
        offset += length
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
    baselines, residual = encode_row_mode_residuals(rows)
    reconstructed = decode_row_mode_residuals((len(row) for row in rows), baselines, residual)
    if reconstructed != rows:
        raise ValueError("row-mode scale reconstruction failed")
    canonical_bytes = len(lzma.compress(canonical, preset=9))
    hierarchy_bytes = len(lzma.compress(baselines + residual, preset=9))
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-row-mode-xor-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": 20769, "compressed_scale_bytes": 17552, "compressed_code_bytes": 3178, "container_overhead_bytes": 39},
        "quantization": {"bits": 2, "scale_count": len(canonical), "scale_layout": "one exact modal FP8 scale baseline per parameter row plus XOR residuals, then LZMA-9", "storage_efficient": False},
        "screen": {"row_count": len(rows), "baseline_bytes": len(baselines), "canonical_lzma_scale_bytes": canonical_bytes, "row_mode_xor_lzma_scale_bytes": hierarchy_bytes, "lower_bound_container_bytes": hierarchy_bytes + 3178 + 39, "delta_from_accepted_container_bytes": hierarchy_bytes + 3178 + 39 - 20769},
        "verification": {"row_mode_residual_stream_reconstructs_exact_scale_stream": True, "source_scale_count_matches_fixed_runtime": len(canonical) == 24726},
        "decision": "rejected-before-runtime",
        "reason": "The exact row-mode baseline hierarchy does not improve the accepted complete container; no runtime, artifact, or exhaustive policy evaluation was created.",
    }


if __name__ == "__main__":
    report = screen()
    output = Path("artifacts/rejected/int2-packed-scalar-fp8-scale-row-mode-xor-lzma-hierarchy-screen-20260926/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], sort_keys=True))
