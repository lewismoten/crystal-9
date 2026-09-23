"""Render trusted Crystal-9 checkpoints as architecture-aware tensor maps."""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from pathlib import Path

import torch

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CELL = 5
_MARGIN = 14
_HEADER_HEIGHT = 36
_GAP = 18
_FONT = {
    "A": ("010", "101", "111", "101", "101"), "B": ("110", "101", "110", "101", "110"), "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"), "E": ("111", "100", "110", "100", "111"), "F": ("111", "100", "110", "100", "100"), "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"), "I": ("111", "010", "010", "010", "111"), "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"), "L": ("100", "100", "100", "100", "111"), "M": ("101", "111", "111", "101", "101"), "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"), "P": ("110", "101", "110", "100", "100"), "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"), "T": ("111", "010", "010", "010", "010"), "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"), "W": ("101", "101", "111", "111", "101", "101"), "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"), "Z": ("111", "001", "010", "100", "111"), "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"), "2": ("110", "001", "111", "100", "111"), "3": ("110", "001", "011", "001", "110"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"), "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"), "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
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


def _shape(tensor: torch.Tensor) -> tuple[int, int]:
    """Use vertical output rows for both matrices and matching 1-D vectors."""
    if tensor.ndim == 1:
        return tensor.numel(), 1
    return int(tensor.shape[0]), int(math.prod(tensor.shape[1:]))


def _render_tensor(rgb: bytearray, width: int, tensor: torch.Tensor, x: int, y: int) -> tuple[int, int]:
    rows, columns = _shape(tensor)
    values = tensor.detach().to(torch.float32).flatten().tolist()
    peak = max((abs(value) for value in values), default=0.0) or 1.0
    for index, value in enumerate(values):
        cx = x + (index % columns) * _CELL
        cy = y + (index // columns) * _CELL
        color = _color(value / peak)
        for dy in range(_CELL - 1):
            for dx in range(_CELL - 1):
                _pixel(rgb, width, cx + dx, cy + dy, color)
    return columns * _CELL, rows * _CELL


def _render_pair(rgb: bytearray, width: int, state: dict[str, torch.Tensor], weight: str, bias: str, label: str, x: int, y: int) -> tuple[int, int]:
    _text(rgb, width, x, y, label, (226, 232, 240))
    grid_y = y + _HEADER_HEIGHT
    _text(rgb, width, x, y + 12, "WEIGHTS", (148, 163, 184))
    weight_width, weight_height = _render_tensor(rgb, width, state[weight], x, grid_y)
    bias_x = x + weight_width + _CELL
    _text(rgb, width, bias_x, y + 12, "BIAS", (148, 163, 184))
    _render_tensor(rgb, width, state[bias], bias_x, grid_y)
    return weight_width + _CELL * 2, max(weight_height, state[bias].numel() * _CELL) + _HEADER_HEIGHT


def _render_single(rgb: bytearray, width: int, state: dict[str, torch.Tensor], name: str, label: str, x: int, y: int) -> tuple[int, int]:
    _text(rgb, width, x, y, label, (226, 232, 240))
    tensor_width, tensor_height = _render_tensor(rgb, width, state[name], x, y + _HEADER_HEIGHT)
    return tensor_width, tensor_height + _HEADER_HEIGHT


def render_checkpoint_inspector(source: Path) -> tuple[bytes, dict[str, object]]:
    """Create a derived architecture map; it is not a byte transport artifact."""
    state = _checkpoint_state(source)
    # Three complete expert columns are the widest section; avoid a fourth empty column.
    width = 800
    height = 2260
    rgb = bytearray(b"\x0b\x10\x18" * (width * height))
    inventory = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", ""),
               "max_abs": max((abs(value) for value in tensor.detach().to(torch.float32).flatten().tolist()), default=0.0)}
        for name, tensor in state.items()
    }

    y = _MARGIN
    _text(rgb, width, _MARGIN, y, "INPUTS", (148, 163, 184))
    _text(rgb, width, 350, y, "BIAS COLUMN", (148, 163, 184))
    _text(rgb, width, 350, y + 12, "MATCHES MATRIX ROWS", (148, 163, 184))
    y += _HEADER_HEIGHT + 12
    _render_single(rgb, width, state, "embedding.weight", "TOKEN_EMBED", _MARGIN, y)
    _render_single(rgb, width, state, "position.weight", "POSITION_EMBED", _MARGIN + 220, y)
    y += 110

    _text(rgb, width, _MARGIN, y, "ATTENTION", (148, 163, 184))
    y += _HEADER_HEIGHT
    _, attention_height = _render_pair(rgb, width, state, "attention.in_proj_weight", "attention.in_proj_bias", "INPUT PROJECTION", _MARGIN, y)
    _render_pair(rgb, width, state, "attention.out_proj.weight", "attention.out_proj.bias", "OUTPUT PROJECTION", 230, y)
    y += attention_height + _GAP

    _text(rgb, width, _MARGIN, y, "NORM_ROUTER_OUTPUT", (148, 163, 184))
    y += _HEADER_HEIGHT
    _render_pair(rgb, width, state, "norm.weight", "norm.bias", "NORM", _MARGIN, y)
    _render_pair(rgb, width, state, "router.weight", "router.bias", "ROUTER", 90, y)
    _render_pair(rgb, width, state, "output.weight", "output.bias", "OUTPUT", 300, y)
    y += 115

    _text(rgb, width, _MARGIN, y, "EXPERTS_0_TO_8", (148, 163, 184))
    y += _HEADER_HEIGHT
    expert_width = 235
    expert_height = 420
    for expert in range(9):
        x = _MARGIN + (expert % 3) * (expert_width + _GAP)
        top = y + (expert // 3) * (expert_height + _GAP)
        _text(rgb, width, x, top, f"EXPERT_{expert}", (148, 163, 184))
        _render_pair(rgb, width, state, f"experts.{expert}.0.weight", f"experts.{expert}.0.bias", "FIRST LAYER", x, top + _HEADER_HEIGHT)
        _render_pair(rgb, width, state, f"experts.{expert}.2.weight", f"experts.{expert}.2.bias", "SECOND LAYER", x, top + 220)

    metadata = {
        "format": "crystal-9-tensor-inspector-v2", "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "tensor_count": len(state),
        "normalization": "per-tensor symmetric max-absolute", "layout": "architecture-grouped-v2",
        "bias_alignment": "vertical output-row axis", "sections": ["inputs", "attention", "norm_router_output", "experts", "output"],
        "legend": {"WEIGHTS": "matrix; rows are output features", "BIAS": "bias column; one value per output row", "B": "bias column; one value per output row"},
        "expert_layout": "3x3 complete expert blocks, layer 1 above layer 2",
        "tensors": inventory, "width": width, "height": height,
    }
    rows = b"".join(b"\0" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    png = b"".join((_SIGNATURE, _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                    _chunk(b"tEXt", b"crystal9-inspector\0" + str(metadata).encode()),
                    _chunk(b"IDAT", zlib.compress(rows, level=9)), _chunk(b"IEND", b"")))
    return png, metadata
