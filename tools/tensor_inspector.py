"""Render trusted Crystal-9 checkpoints as architecture-aware tensor maps."""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from pathlib import Path

import torch
from PIL import ImageFont

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CELL = 5
_MARGIN = 20
_HEADER_HEIGHT = 48
_TEXT_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
_LABEL = (226, 232, 240)
_MUTED = (148, 163, 184)
_FLOW = (250, 204, 21)
_WEIGHT_BORDER = (45, 212, 191)
_BIAS_BORDER = (251, 146, 60)
_VOCABULARY = ["<pad>", "<bos>", "<eos>", "!", "a", "b", "c", "d", "e", "f", "g", "h", "i"]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _pixel(rgb: bytearray, width: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < len(rgb) // (width * 3):
        rgb[(y * width + x) * 3 : (y * width + x + 1) * 3] = bytes(color)


def _text(rgb: bytearray, width: int, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    """Render anti-aliased DejaVu Sans into the existing RGB canvas."""
    mask = _TEXT_FONT.getmask(text, mode="L")
    text_width, text_height = mask.size
    for dy in range(text_height):
        for dx in range(text_width):
            alpha = mask.getpixel((dx, dy))
            if alpha:
                offset_x, offset_y = x + dx, y + dy
                if 0 <= offset_x < width and 0 <= offset_y < len(rgb) // (width * 3):
                    start = (offset_y * width + offset_x) * 3
                    prior = rgb[start : start + 3]
                    rgb[start : start + 3] = bytes(
                        round(prior[channel] * (255 - alpha) / 255 + color[channel] * alpha / 255)
                        for channel in range(3)
                    )


def _centered_text(rgb: bytearray, width: int, center_x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    left, _, right, _ = _TEXT_FONT.getbbox(text)
    _text(rgb, width, center_x - (right - left) // 2, y, text, color)


def _rectangle(rgb: bytearray, width: int, x: int, y: int, box_width: int, box_height: int, color: tuple[int, int, int], thickness: int = 2) -> None:
    for offset in range(thickness):
        for px in range(x - offset, x + box_width + offset + 1):
            _pixel(rgb, width, px, y - offset, color)
            _pixel(rgb, width, px, y + box_height + offset, color)
        for py in range(y - offset, y + box_height + offset + 1):
            _pixel(rgb, width, x - offset, py, color)
            _pixel(rgb, width, x + box_width + offset, py, color)


def _ellipse(rgb: bytearray, width: int, center_x: int, center_y: int, radius_x: int, radius_y: int) -> None:
    for py in range(center_y - radius_y, center_y + radius_y + 1):
        for px in range(center_x - radius_x, center_x + radius_x + 1):
            value = ((px - center_x) / radius_x) ** 2 + ((py - center_y) / radius_y) ** 2
            if value <= 1:
                _pixel(rgb, width, px, py, (11, 16, 24))
            if 0.90 <= value <= 1.10:
                _pixel(rgb, width, px, py, _FLOW)


def _color(value: float) -> tuple[int, int, int]:
    strength = min(abs(value), 1.0)
    base = int(18 + 222 * strength)
    return (22, int(35 + 95 * (1 - strength)), base) if value < 0 else (base, int(45 + 175 * strength), 26)


def _checkpoint_state(source: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"checkpoint has no tensor state_dict: {source}")
    return state


def _shape(tensor: torch.Tensor) -> tuple[int, int]:
    """Matrices and matching bias vectors share a vertical output-row axis."""
    return (tensor.numel(), 1) if tensor.ndim == 1 else (int(tensor.shape[0]), int(math.prod(tensor.shape[1:])))


def _render_tensor(rgb: bytearray, width: int, tensor: torch.Tensor, x: int, y: int) -> tuple[int, int]:
    rows, columns = _shape(tensor)
    values = tensor.detach().to(torch.float32).flatten().tolist()
    peak = max((abs(value) for value in values), default=0.0) or 1.0
    for index, value in enumerate(values):
        cx, cy = x + (index % columns) * _CELL, y + (index // columns) * _CELL
        color = _color(value / peak)
        for dy in range(_CELL - 1):
            for dx in range(_CELL - 1):
                _pixel(rgb, width, cx + dx, cy + dy, color)
    _rectangle(rgb, width, x, y, columns * _CELL - 1, rows * _CELL - 1, _BIAS_BORDER if tensor.ndim == 1 else _WEIGHT_BORDER)
    return columns * _CELL, rows * _CELL


def _line(rgb: bytearray, width: int, x1: int, y1: int, x2: int, y2: int) -> None:
    if x1 == x2:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            for dx in (-1, 0, 1):
                _pixel(rgb, width, x1 + dx, y, _FLOW)
    elif y1 == y2:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for dy in (-1, 0, 1):
                _pixel(rgb, width, x, y1 + dy, _FLOW)
    else:
        raise ValueError("connectors must be horizontal or vertical")


def _arrow(rgb: bytearray, width: int, x1: int, y1: int, x2: int, y2: int) -> None:
    """Draw an arrow whose wings trail behind the destination tip."""
    if y1 == y2:
        direction = 1 if x2 > x1 else -1
        _line(rgb, width, x1, y1, x2 - direction * 8, y1)
        for offset in range(8):
            _pixel(rgb, width, x2 - direction * offset, y2 - offset, _FLOW)
            _pixel(rgb, width, x2 - direction * offset, y2 + offset, _FLOW)
    elif x1 == x2:
        direction = 1 if y2 > y1 else -1
        _line(rgb, width, x1, y1, x1, y2 - direction * 8)
        for offset in range(8):
            _pixel(rgb, width, x2 - offset, y2 - direction * offset, _FLOW)
            _pixel(rgb, width, x2 + offset, y2 - direction * offset, _FLOW)
    else:
        raise ValueError("arrows must be horizontal or vertical")


def _render_pair(rgb: bytearray, width: int, state: dict[str, torch.Tensor], weight: str, bias: str, label: str, x: int, y: int) -> tuple[int, int]:
    _text(rgb, width, x, y, label, _LABEL)
    grid_y = y + _HEADER_HEIGHT
    weight_width, weight_height = _render_tensor(rgb, width, state[weight], x, grid_y)
    bias_x = x + weight_width + _CELL
    _render_tensor(rgb, width, state[bias], bias_x, grid_y)
    return weight_width + _CELL * 2, max(weight_height, state[bias].numel() * _CELL) + _HEADER_HEIGHT


def _render_single(rgb: bytearray, width: int, state: dict[str, torch.Tensor], name: str, label: str, x: int, y: int) -> tuple[int, int]:
    _text(rgb, width, x, y, label, _LABEL)
    tensor_width, tensor_height = _render_tensor(rgb, width, state[name], x, y + _HEADER_HEIGHT)
    return tensor_width, tensor_height + _HEADER_HEIGHT


def _render_vocabulary(rgb: bytearray, width: int, x: int, y: int) -> None:
    _text(rgb, width, x, y, "Vocabulary token IDs", _LABEL)
    for token_id, token in enumerate(_VOCABULARY):
        token_name = {"<pad>": "PAD", "<bos>": "BOS", "<eos>": "EOS", "!": "INVALID"}.get(token, token.upper())
        _text(rgb, width, x, y + 26 + token_id * 22, f"{token_id:02d}  {token_name}", _MUTED)


def render_checkpoint_inspector(source: Path) -> tuple[bytes, dict[str, object]]:
    """Create a derived, architecture-flow map; it is not a byte transport artifact."""
    state = _checkpoint_state(source)
    width, height = 2050, 1390
    rgb = bytearray(b"\x0b\x10\x18" * (width * height))
    inventory = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", ""),
               "max_abs": max((abs(value) for value in tensor.detach().to(torch.float32).flatten().tolist()), default=0.0)}
        for name, tensor in state.items()
    }

    top_y, flow_y = 100, 70
    _text(rgb, width, _MARGIN, 25, "Model flow left to right", _MUTED)
    _text(rgb, width, 1540, 25, "Borders: teal weights | orange bias", _MUTED)
    _render_single(rgb, width, state, "embedding.weight", "Token embedding", 20, top_y)
    _render_single(rgb, width, state, "position.weight", "Position embedding", 210, top_y)
    _render_pair(rgb, width, state, "attention.in_proj_weight", "attention.in_proj_bias", "Input projection", 400, top_y)
    _render_pair(rgb, width, state, "attention.out_proj.weight", "attention.out_proj.bias", "Output projection", 630, top_y)
    _render_pair(rgb, width, state, "norm.weight", "norm.bias", "Norm", 850, top_y)
    _render_pair(rgb, width, state, "router.weight", "router.bias", "Router", 950, top_y)
    _render_pair(rgb, width, state, "output.weight", "output.bias", "Final output", 1160, top_y)
    _render_vocabulary(rgb, width, 20, 230)

    _arrow(rgb, width, 175, flow_y, 395, flow_y)
    _arrow(rgb, width, 605, flow_y, 625, flow_y)
    _arrow(rgb, width, 825, flow_y, 845, flow_y)
    _arrow(rgb, width, 925, flow_y, 945, flow_y)

    expert_y, expert_gap, expert_width = 720, 16, 200
    router_center, router_bus_y = 1030, expert_y - 34
    expert_left, expert_right = 12, 1942
    expert_top, expert_bottom = expert_y - 42, 1270
    _rectangle(rgb, width, expert_left, expert_top, expert_right - expert_left, expert_bottom - expert_top, _FLOW)
    _text(rgb, width, expert_left + 16, expert_top + 12, "Experts", _LABEL)
    first_top, first_bottom = expert_y + 16, expert_y + 265
    second_top, second_bottom = expert_y + 300, expert_y + 530
    _rectangle(rgb, width, expert_left + 12, first_top, expert_right - expert_left - 24, first_bottom - first_top, _MUTED, 1)
    _rectangle(rgb, width, expert_left + 12, second_top, expert_right - expert_left - 24, second_bottom - second_top, _MUTED, 1)
    _text(rgb, width, expert_left + 28, first_top + 8, "First layer", _LABEL)
    _text(rgb, width, expert_left + 28, second_top + 8, "Second layer", _LABEL)
    _arrow(rgb, width, router_center, 205, router_center, 395)
    _line(rgb, width, router_center, 485, router_center, expert_top - 8)
    _ellipse(rgb, width, router_center, 440, 112, 38)
    _centered_text(rgb, width, router_center, 429, "Selected: 2 experts", _LABEL)
    for expert in range(9):
        x = _MARGIN + expert * (expert_width + expert_gap)
        _render_pair(rgb, width, state, f"experts.{expert}.0.weight", f"experts.{expert}.0.bias", "", x, expert_y + 48)
        _render_pair(rgb, width, state, f"experts.{expert}.2.weight", f"experts.{expert}.2.bias", "", x, expert_y + 298)
    _arrow(rgb, width, (expert_left + expert_right) // 2, first_bottom + 8, (expert_left + expert_right) // 2, second_top - 8)
    # The output panel shares the router's top-row alignment; the final path rises from the complete experts box.
    _line(rgb, width, expert_right, (expert_top + expert_bottom) // 2, expert_right, 70)
    _arrow(rgb, width, expert_right, 70, 1155, 70)

    metadata = {
        "format": "crystal-9-tensor-inspector-v6", "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "tensor_count": len(state),
        "normalization": "per-tensor symmetric max-absolute", "layout": "architecture-flow-v6",
        "bias_alignment": "vertical output-row axis", "sections": ["inputs", "attention", "norm_router", "experts", "output"],
        "legend": {"WEIGHTS": "matrix; rows are output features", "BIAS": "bias column; one value per output row", "B": "bias column; one value per output row"},
        "expert_layout": "one Experts box with First layer and Second layer boxed within it",
        "calculation_flow": ["embeddings and positions", "attention", "norm and router", "top-2 routed experts", "combined output"],
        "vocabulary": _VOCABULARY, "font": "DejaVu Sans", "router_selection_label": "Selected: 2 experts",
        "border_legend": {"teal": "weight matrix", "orange": "bias vector"}, "tensors": inventory, "width": width, "height": height,
    }
    rows = b"".join(b"\0" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    png = b"".join((_SIGNATURE, _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                    _chunk(b"tEXt", b"crystal9-inspector\0" + str(metadata).encode()),
                    _chunk(b"IDAT", zlib.compress(rows, level=9)), _chunk(b"IEND", b"")))
    return png, metadata
