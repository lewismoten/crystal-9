"""Render trusted Crystal-9 checkpoints as tensor-aware PNG inspection maps."""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from pathlib import Path

import torch

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CELL = 5
_PANEL_WIDTH = 244
_PANEL_GAP = 12
_COLUMNS = 4
_MARGIN = 14
_HEADER_HEIGHT = 24
_FONT = {
    "A": ("010", "101", "111", "101", "101"), "B": ("110", "101", "110", "101", "110"), "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"), "E": ("111", "100", "110", "100", "111"), "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"), "I": ("111", "010", "010", "010", "111"), "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"), "M": ("101", "111", "111", "101", "101"), "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"), "P": ("110", "101", "110", "100", "100"), "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"), "T": ("111", "010", "010", "010", "010"), "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"), "W": ("101", "101", "111", "111", "101"), "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"), "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "111", "100", "111"), "3": ("110", "001", "011", "001", "110"), "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"), "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"), ".": ("000", "000", "000", "000", "010"),
    "_": ("000", "000", "000", "000", "111"), " ": ("000", "000", "000", "000", "000"),
}


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _pixel(rgb: bytearray, width: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < len(rgb) // (width * 3):
        rgb[(y * width + x) * 3 : (y * width + x + 1) * 3] = bytes(color)


def _text(rgb: bytearray, width: int, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    for index, char in enumerate(text.upper()):
        glyph = _FONT.get(char, _FONT[" "])
        left = x + index * 8
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    for dy in range(2):
                        for dx in range(2):
                            _pixel(rgb, width, left + column * 2 + dx, y + row * 2 + dy, color)


def _color(value: float) -> tuple[int, int, int]:
    strength = min(abs(value), 1.0)
    base = int(18 + 222 * strength)
    if value < 0:
        return (22, int(35 + 95 * (1 - strength)), base)
    return (base, int(45 + 175 * strength), 26)


def _checkpoint_state(source: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"checkpoint has no tensor state_dict: {source}")
    return state


def render_checkpoint_inspector(source: Path) -> tuple[bytes, dict[str, object]]:
    """Create a derived map of actual checkpoint tensors; it is not byte transport."""
    state = _checkpoint_state(source)
    panels: list[tuple[str, torch.Tensor, list[int], float, int, int]] = []
    for name, tensor in state.items():
        values = tensor.detach().to(torch.float32).flatten().tolist()
        shape = list(tensor.shape)
        rows = shape[0] if len(shape) > 1 else 1
        columns = int(math.ceil(len(values) / rows))
        peak = max((abs(value) for value in values), default=0.0) or 1.0
        panels.append((name, tensor, shape, peak, rows, columns))
    row_heights = []
    for start in range(0, len(panels), _COLUMNS):
        row_heights.append(max(_HEADER_HEIGHT + rows * _CELL + 12 for _, _, _, _, rows, _ in panels[start : start + _COLUMNS]))
    width = _MARGIN * 2 + _COLUMNS * _PANEL_WIDTH + (_COLUMNS - 1) * _PANEL_GAP
    height = _MARGIN * 2 + sum(row_heights) + _PANEL_GAP * (len(row_heights) - 1)
    rgb = bytearray(b"\x0b\x10\x18" * (width * height))
    inventory: dict[str, dict[str, object]] = {}
    y = _MARGIN
    for group, row_height in zip(range(0, len(panels), _COLUMNS), row_heights):
        for column, (name, tensor, shape, peak, rows, columns) in enumerate(panels[group : group + _COLUMNS]):
            x = _MARGIN + column * (_PANEL_WIDTH + _PANEL_GAP)
            _text(rgb, width, x, y + 4, name[:30], (226, 232, 240))
            values = tensor.detach().to(torch.float32).flatten().tolist()
            for index, value in enumerate(values):
                cx = x + (index % columns) * _CELL
                cy = y + _HEADER_HEIGHT + (index // columns) * _CELL
                color = _color(value / peak)
                for dy in range(_CELL - 1):
                    for dx in range(_CELL - 1):
                        _pixel(rgb, width, cx + dx, cy + dy, color)
            inventory[name] = {"shape": shape, "dtype": str(tensor.dtype).replace("torch.", ""), "max_abs": peak}
        y += row_height + _PANEL_GAP
    rows = b"".join(b"\0" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    metadata = {
        "format": "crystal-9-tensor-inspector-v1", "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "tensor_count": len(state),
        "normalization": "per-tensor symmetric max-absolute", "tensors": inventory,
        "width": width, "height": height,
    }
    png = b"".join((_SIGNATURE, _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                    _chunk(b"tEXt", b"crystal9-inspector\0" + str(metadata).encode()),
                    _chunk(b"IDAT", zlib.compress(rows, level=9)), _chunk(b"IEND", b"")))
    return png, metadata
