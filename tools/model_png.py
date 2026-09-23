"""Lossless truecolor PNG transport containing only Crystal-9 payload pixels."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_rgba_png(payload: bytes, *, model_name: str = "Crystal-9", precision: str = "unknown") -> bytes:
    """Encode artifact bytes in RGB pixels; the legacy function name is retained."""
    if not payload:
        raise ValueError("payload must not be empty")
    pixels = math.ceil(len(payload) / 3)
    width = math.ceil(math.sqrt(pixels))
    height = math.ceil(pixels / width)
    rgb = bytearray(width * height * 3)
    rgb[: len(payload)] = payload
    rows = b"".join(b"\0" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    digest = hashlib.sha256(payload).hexdigest()
    metadata = json.dumps(
        {"format": "crystal-9-rgb-byte-png-v4", "source_bytes": len(payload), "sha256": digest,
         "model_name": model_name, "precision": precision, "data_height": height, "footer_height": 0},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return b"".join((_SIGNATURE, _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                     _chunk(b"tEXt", b"crystal9\0" + metadata), _chunk(b"IDAT", zlib.compress(rows, level=9)), _chunk(b"IEND", b"")))


def decode_rgba_png(png: bytes) -> tuple[bytes, dict[str, object]]:
    """Decode the fixed RGB contract and verify the PNG-comment SHA-256."""
    if not png.startswith(_SIGNATURE):
        raise ValueError("not a PNG")
    offset = len(_SIGNATURE); chunks: list[tuple[bytes, bytes]] = []
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]; data = png[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != crc:
            raise ValueError("PNG CRC mismatch")
        chunks.append((kind, data)); offset += 12 + length
        if kind == b"IEND":
            break
    ihdr = next(data for kind, data in chunks if kind == b"IHDR")
    width, height, depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", ihdr)
    if (depth, color_type, compression, filter_method, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("unsupported PNG encoding")
    text = next(data for kind, data in chunks if kind == b"tEXt" and data.startswith(b"crystal9\0"))
    metadata = json.loads(text.split(b"\0", 1)[1])
    if metadata.get("format") != "crystal-9-rgb-byte-png-v4" or int(metadata["footer_height"]) != 0:
        raise ValueError("unsupported Crystal-9 PNG format")
    if height != int(metadata["data_height"]):
        raise ValueError("unexpected payload height")
    raw = zlib.decompress(b"".join(data for kind, data in chunks if kind == b"IDAT")); stride = width * 3
    if len(raw) != height * (stride + 1) or any(raw[row * (stride + 1)] != 0 for row in range(height)):
        raise ValueError("unexpected PNG row filter")
    rgb = b"".join(raw[row * (stride + 1) + 1 : (row + 1) * (stride + 1)] for row in range(height))
    payload = rgb[: int(metadata["source_bytes"])]
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("payload SHA-256 mismatch")
    metadata.update({"width": width, "height": height})
    return payload, metadata
