"""Lossless RGB PNG byte transport with visible Crystal-9 provenance footer."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_FOOTER_HEIGHT = 18
_MIN_FOOTER_WIDTH = 320
_FONT = {
    "A": ("010", "101", "111", "101", "101"), "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"), "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"), "F": ("111", "100", "110", "100", "100"),
    "H": ("101", "101", "111", "101", "101"), "I": ("111", "010", "010", "010", "111"),
    "K": ("101", "101", "110", "101", "101"), "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"), "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"), "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "011", "001"), "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"), "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"), "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"), "Z": ("111", "001", "010", "100", "111"),
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "111", "100", "111"), "3": ("110", "001", "011", "001", "110"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
    ":": ("000", "010", "000", "010", "000"), "-": ("000", "000", "111", "000", "000"),
    "|": ("010", "010", "010", "010", "010"), " ": ("000", "000", "000", "000", "000"),
}


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _draw_text(rgb: bytearray, width: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    for index, char in enumerate(text.upper()):
        glyph = _FONT.get(char, _FONT[" "])
        x = 3 + index * 4
        if x + 3 > width:
            break
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    offset = ((y + row) * width + x + column) * 3
                    rgb[offset : offset + 3] = bytes(color)


def encode_rgba_png(payload: bytes, *, model_name: str = "Crystal-9", precision: str = "unknown") -> bytes:
    """Encode payload as truecolor RGB; the historical name remains compatible."""
    if not payload:
        raise ValueError("payload must not be empty")
    pixels = math.ceil(len(payload) / 3)
    width = max(math.ceil(math.sqrt(pixels)), _MIN_FOOTER_WIDTH)
    data_height = math.ceil(pixels / width)
    height = data_height + _FOOTER_HEIGHT
    rgb = bytearray(width * height * 3)
    for pixel, start in enumerate(range(0, len(payload), 3)):
        chunk = payload[start : start + 3]
        rgb[pixel * 3 : pixel * 3 + 3] = chunk + b"\0" * (3 - len(chunk))
    footer_offset = data_height * width * 3
    rgb[footer_offset:] = b"\x0b\x10\x18" * (width * _FOOTER_HEIGHT)
    digest = hashlib.sha256(payload).hexdigest()
    _draw_text(rgb, width, data_height + 3, f"MODEL:{model_name} | PRECISION:{precision}", (217, 249, 157))
    _draw_text(rgb, width, data_height + 10, f"SHA256:{digest}", (125, 211, 252))
    rows = b"".join(b"\0" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    metadata = json.dumps(
        {"format": "crystal-9-rgb-byte-png-v2", "source_bytes": len(payload), "sha256": digest, "model_name": model_name,
         "precision": precision, "data_height": data_height, "footer_height": _FOOTER_HEIGHT},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return b"".join((_SIGNATURE, _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                     _chunk(b"tEXt", b"crystal9\0" + metadata), _chunk(b"IDAT", zlib.compress(rows, level=9)), _chunk(b"IEND", b"")))


def decode_rgba_png(png: bytes) -> tuple[bytes, dict[str, object]]:
    """Decode the fixed Crystal-9 RGB contract and verify its payload digest."""
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
        if kind == b"IEND": break
    ihdr = next(data for kind, data in chunks if kind == b"IHDR")
    width, height, depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", ihdr)
    if (depth, color_type, compression, filter_method, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("unsupported PNG encoding")
    text = next(data for kind, data in chunks if kind == b"tEXt" and data.startswith(b"crystal9\0"))
    metadata = json.loads(text.split(b"\0", 1)[1])
    data_height = int(metadata["data_height"])
    if height != data_height + int(metadata["footer_height"]):
        raise ValueError("footer layout mismatch")
    raw = zlib.decompress(b"".join(data for kind, data in chunks if kind == b"IDAT")); stride = width * 3
    if len(raw) != height * (stride + 1) or any(raw[row * (stride + 1)] != 0 for row in range(height)):
        raise ValueError("unexpected PNG row filter")
    rgb = b"".join(raw[row * (stride + 1) + 1 : (row + 1) * (stride + 1)] for row in range(data_height))
    payload = rgb[: int(metadata["source_bytes"])]
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("payload SHA-256 mismatch")
    metadata.update({"width": width, "height": height})
    return payload, metadata
