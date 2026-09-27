"""Exact packed FP8 bitfield scale-hierarchy screen."""

from __future__ import annotations

import hashlib
import json
import lzma
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal9 import TinyMoEPolicy


def _pack_fixed_width(values: bytes, width: int) -> bytes:
    accumulator = 0
    bits = 0
    output = bytearray()
    for value in values:
        accumulator |= value << bits
        bits += width
        while bits >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            bits -= 8
    if bits:
        output.append(accumulator)
    return bytes(output)


def _unpack_fixed_width(payload: bytes, count: int, width: int) -> bytes:
    accumulator = 0
    bits = 0
    offset = 0
    values = bytearray()
    mask = (1 << width) - 1
    while len(values) < count:
        while bits < width:
            if offset == len(payload):
                raise ValueError("component payload is truncated")
            accumulator |= payload[offset] << bits
            offset += 1
            bits += 8
        values.append(accumulator & mask)
        accumulator >>= width
        bits -= width
    return bytes(values)


def encode_fp8_bitfields(scale_bytes: bytes) -> tuple[bytes, bytes]:
    """Split E4M3FN bytes into exact packed sign/exponent and mantissa streams."""
    return (
        bytes(value >> 3 for value in scale_bytes),
        _pack_fixed_width(bytes(value & 0x07 for value in scale_bytes), 3),
    )


def decode_fp8_bitfields(upper: bytes, mantissas: bytes) -> bytes:
    expected_mantissa_bytes = (len(upper) * 3 + 7) // 8
    if len(mantissas) != expected_mantissa_bytes:
        raise ValueError("bitfield component length does not match scale count")
    lower = _unpack_fixed_width(mantissas, len(upper), 3)
    return bytes((high << 3) | low for high, low in zip(upper, lower))


def _scale_bytes(source: TinyMoEPolicy) -> bytes:
    return b"".join(
        value.detach().abs().to(torch.float8_e4m3fn).view(torch.uint8).contiguous().numpy().tobytes()
        for value in source.state_dict().values()
    )


def screen() -> dict:
    checkpoint_path = Path("artifacts-fp32.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size=13).eval()
    source.load_state_dict(checkpoint["state_dict"])
    canonical = _scale_bytes(source)
    upper, mantissas = encode_fp8_bitfields(canonical)
    if decode_fp8_bitfields(upper, mantissas) != canonical:
        raise ValueError("packed bitfield scale reconstruction failed")
    payload = upper + mantissas
    canonical_bytes = len(lzma.compress(canonical, preset=9))
    hierarchy_bytes = len(lzma.compress(payload, preset=9))
    return {
        "layout": "complete-scalar-group-int2-packed-fp8-scale-packed-bitfield-lzma-hierarchy-screen",
        "source_checkpoint": "artifacts-fp32.pt",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "accepted_predecessor": {"artifact_bytes": 20769, "compressed_scale_bytes": 17552, "compressed_code_bytes": 3178, "container_overhead_bytes": 39},
        "quantization": {"bits": 2, "scale_count": len(canonical), "scale_layout": "lossless packed 5-bit sign/exponent stream plus packed 3-bit mantissa stream, then LZMA-9", "storage_efficient": False},
        "screen": {"upper_raw_bytes": len(upper), "mantissa_raw_bytes": len(mantissas), "canonical_lzma_scale_bytes": canonical_bytes, "packed_bitfield_lzma_scale_bytes": hierarchy_bytes, "lower_bound_container_bytes": hierarchy_bytes + 3178 + 39, "delta_from_accepted_container_bytes": hierarchy_bytes + 3178 + 39 - 20769},
        "verification": {"packed_bitfield_stream_reconstructs_exact_scale_stream": True, "source_scale_count_matches_fixed_runtime": len(canonical) == 24726},
        "decision": "rejected-before-runtime" if hierarchy_bytes >= 17552 else "advance-to-runtime",
        "reason": "The exact packed-bitfield hierarchy does not improve the accepted complete container; no runtime, artifact, or exhaustive policy evaluation was created." if hierarchy_bytes >= 17552 else "The exact packed-bitfield hierarchy improves the accepted scale stream and requires a separately parity-tested runtime candidate.",
    }


if __name__ == "__main__":
    report = screen()
    output = Path("artifacts/rejected/int2-packed-scalar-fp8-scale-packed-bitfield-lzma-hierarchy-screen-20260927/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], sort_keys=True))
