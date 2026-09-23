"""Lossless RGBA PNG byte transport for Crystal-9 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_rgba_png(payload: bytes) -> bytes:
    """Encode bytes as opaque RGBA pixels with lossless, explicit metadata."""
    if not payload:
        raise ValueError("payload must not be empty")
    pixels = math.ceil(len(payload) / 3)
    width = math.ceil(math.sqrt(pixels))
    height = math.ceil(pixels / width)
    rgba = bytearray(width * height * 4)
    for pixel, start in enumerate(range(0, len(payload), 3)):
        chunk = payload[start : start + 3]
        rgba[pixel * 4 : pixel * 4 + 3] = chunk + b"\0" * (3 - len(chunk))
        rgba[pixel * 4 + 3] = 255
    for pixel in range(math.ceil(len(payload) / 3), width * height):
        rgba[pixel * 4 + 3] = 255
    rows = b"".join(b"\0" + rgba[row * width * 4 : (row + 1) * width * 4] for row in range(height))
    metadata = json.dumps(
        {"format": "crystal-9-rgba-byte-png-v1", "source_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return b"".join(
        (
            _SIGNATURE,
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)),
            _chunk(b"tEXt", b"crystal9\0" + metadata),
            _chunk(b"IDAT", zlib.compress(rows, level=9)),
            _chunk(b"IEND", b""),
        )
    )


def decode_rgba_png(png: bytes) -> tuple[bytes, dict[str, object]]:
    """Decode only the fixed Crystal-9 RGBA PNG contract and verify its digest."""
    if not png.startswith(_SIGNATURE):
        raise ValueError("not a PNG")
    offset = len(_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]
        data = png[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != crc:
            raise ValueError("PNG CRC mismatch")
        chunks.append((kind, data))
        offset += 12 + length
        if kind == b"IEND":
            break
    ihdr = next(data for kind, data in chunks if kind == b"IHDR")
    width, height, depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", ihdr)
    if (depth, color_type, compression, filter_method, interlace) != (8, 6, 0, 0, 0):
        raise ValueError("unsupported PNG encoding")
    text = next(data for kind, data in chunks if kind == b"tEXt" and data.startswith(b"crystal9\0"))
    metadata = json.loads(text.split(b"\0", 1)[1])
    raw = zlib.decompress(b"".join(data for kind, data in chunks if kind == b"IDAT"))
    stride = width * 4
    if len(raw) != height * (stride + 1) or any(raw[row * (stride + 1)] != 0 for row in range(height)):
        raise ValueError("unexpected PNG row filter")
    rgba = b"".join(raw[row * (stride + 1) + 1 : (row + 1) * (stride + 1)] for row in range(height))
    source_bytes = int(metadata["source_bytes"])
    payload = b"".join(rgba[index : index + 3] for index in range(0, len(rgba), 4))[:source_bytes]
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("payload SHA-256 mismatch")
    metadata.update({"width": width, "height": height})
    return payload, metadata
