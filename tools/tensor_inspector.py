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
_EXPERT_BORDER = (100, 116, 139)
_WEIGHT_BORDER = (118, 92, 160)
_BIAS_BORDER = (64, 170, 180)
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


def _fill_rectangle(rgb: bytearray, width: int, x: int, y: int, box_width: int, box_height: int, color: tuple[int, int, int]) -> None:
    for py in range(y, y + box_height):
        for px in range(x, x + box_width):
            _pixel(rgb, width, px, py, color)


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
    if value < 0:
        return (22, int(35 + 95 * (1 - strength)), int(18 + 222 * strength))
    # Keep moderate positive values visibly green before high positive values transition to yellow.
    return (int(18 + 222 * strength ** 3), int(60 + 160 * strength), 26)


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
    weight_columns = _shape(state[weight])[1]
    total_width = weight_columns * _CELL + _CELL * 2
    if label:
        _centered_text(rgb, width, x + total_width // 2, y + 20, label, _LABEL)
    grid_y = y + _HEADER_HEIGHT
    weight_width, weight_height = _render_tensor(rgb, width, state[weight], x, grid_y)
    bias_x = x + weight_width + _CELL
    _render_tensor(rgb, width, state[bias], bias_x, grid_y)
    return weight_width + _CELL * 2, max(weight_height, state[bias].numel() * _CELL) + _HEADER_HEIGHT


def _render_attention_input_projections(rgb: bytearray, width: int, state: dict[str, torch.Tensor], x: int, y: int) -> None:
    """Render PyTorch's packed Q/K/V input projection as three logical pairs."""
    group_x, group_y, group_width, group_height = x - 15, y, 210, 650
    _rectangle(rgb, width, group_x, group_y, group_width, group_height, _EXPERT_BORDER, 1)
    _centered_text(rgb, width, group_x + group_width // 2, group_y + 10, "Attention input", _LABEL)
    _centered_text(rgb, width, group_x + group_width // 2, group_y + 32, "projections", _LABEL)
    packed_weight = state["attention.in_proj_weight"]
    packed_bias = state["attention.in_proj_bias"]
    for index, meaning in enumerate(("Query", "Key", "Value")):
        label_y = group_y + 55 + index * 196
        grid_y = label_y + 27
        _centered_text(rgb, width, x + 85, label_y, meaning, _LABEL)
        weight = packed_weight[index * 32 : (index + 1) * 32]
        bias = packed_bias[index * 32 : (index + 1) * 32]
        weight_width, _ = _render_tensor(rgb, width, weight, x, grid_y)
        _render_tensor(rgb, width, bias, x + weight_width + _CELL, grid_y)


def _render_single(rgb: bytearray, width: int, state: dict[str, torch.Tensor], name: str, label: str, x: int, y: int) -> tuple[int, int]:
    tensor_columns = _shape(state[name])[1]
    _centered_text(rgb, width, x + tensor_columns * _CELL // 2, y + 20, label, _LABEL)
    tensor_width, tensor_height = _render_tensor(rgb, width, state[name], x, y + _HEADER_HEIGHT)
    return tensor_width, tensor_height + _HEADER_HEIGHT


def _render_vocabulary(rgb: bytearray, width: int, x: int, y: int) -> None:
    _text(rgb, width, x, y, "Vocabulary token IDs", _LABEL)
    for token_id, token in enumerate(_VOCABULARY):
        token_name = {"<pad>": "PAD", "<bos>": "BOS", "<eos>": "EOS", "!": "INVALID"}.get(token, token.upper())
        _text(rgb, width, x, y + 26 + token_id * 22, f"{token_id:02d}  {token_name}", _MUTED)


def _render_execution_contract(rgb: bytearray, width: int, x: int, y: int) -> None:
    """Render the runtime facts needed to interpret this derived view honestly."""
    panel_width, panel_height = 570, 470
    _rectangle(rgb, width, x, y, panel_width, panel_height, _EXPERT_BORDER, 1)
    lines = (
        ("Decoded inspector — not reconstructable", _LABEL),
        ("Proposed release: lewismoten/crystal-9:q4", _FLOW),
        ("", _LABEL),
        ("INPUT / SEQUENCE", _LABEL),
        ("Public input: a-i; maximum 8 moves", _MUTED),
        ("BOS + history; PAD to 9 positions", _MUTED),
        ("Causal attention; read final non-PAD state", _MUTED),
        ("", _LABEL),
        ("MOE / OUTPUT", _LABEL),
        ("Softmax router selects top 2 of 9 experts", _MUTED),
        ("Expert: 32 -> 32, SiLU, 32", _MUTED),
        ("13 logits; public a-i, ! invalid/no-move", _MUTED),
        ("", _LABEL),
        ("SHAPES: embed 13x32; position 9x32; router 9x32", _MUTED),
        ("attention 96x32 / 32x32; expert matrices 32x32", _MUTED),
        ("final output 13x32; norm vector 32", _MUTED),
    )
    for index, (line, color) in enumerate(lines):
        if line:
            _text(rgb, width, x + 16, y + 14 + index * 27, line, color)


def render_checkpoint_inspector(source: Path) -> tuple[bytes, dict[str, object]]:
    """Create a derived, architecture-flow map; it is not a byte transport artifact."""
    state = _checkpoint_state(source)
    width, height = 1640, 1240
    rgb = bytearray(b"\x0b\x10\x18" * (width * height))
    inventory = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", ""),
               "max_abs": max((abs(value) for value in tensor.detach().to(torch.float32).flatten().tolist()), default=0.0)}
        for name, tensor in state.items()
    }

    top_y, flow_y = 40, 110
    _render_single(rgb, width, state, "embedding.weight", "Token embedding", 20, top_y)
    _render_single(rgb, width, state, "position.weight", "Position embedding", 210, top_y)
    _render_attention_input_projections(rgb, width, state, 400, 30)
    _render_pair(rgb, width, state, "attention.out_proj.weight", "attention.out_proj.bias", "Output projection", 630, top_y)
    _render_pair(rgb, width, state, "norm.weight", "norm.bias", "Norm", 850, top_y)
    _render_pair(rgb, width, state, "router.weight", "router.bias", "Router", 950, top_y)
    _render_pair(rgb, width, state, "output.weight", "output.bias", "Final output", 1160, top_y)
    _render_vocabulary(rgb, width, 20, 170)
    _render_execution_contract(rgb, width, 20, 750)

    # Token and position embeddings are combined elementwise before attention.
    _line(rgb, width, 190, flow_y, 205, flow_y)
    _line(rgb, width, 198, flow_y - 8, 198, flow_y + 8)
    _arrow(rgb, width, 375, flow_y, 384, flow_y)
    _arrow(rgb, width, 600, flow_y, 625, flow_y)
    _arrow(rgb, width, 815, flow_y, 835, flow_y)
    _arrow(rgb, width, 898, flow_y, 918, flow_y)

    # Begin below Output projection, then use a 5+4 grid so every expert can contain its full weight+bias pair.
    expert_y, expert_gap, expert_width = 340, 10, 180
    router_center = 1030
    expert_left, expert_right = 630, 1610
    expert_top, expert_bottom = 298, 1223
    _rectangle(rgb, width, expert_left, expert_top, expert_right - expert_left, expert_bottom - expert_top, _EXPERT_BORDER)
    _text(rgb, width, expert_left + 16, expert_top + 12, "Experts", _LABEL)
    _arrow(rgb, width, router_center, 145, router_center, expert_top - 1)
    selection_x, selection_y, selection_width, selection_height = router_center - 92, 178, 184, 76
    _fill_rectangle(rgb, width, selection_x, selection_y, selection_width, selection_height, (11, 16, 24))
    _rectangle(rgb, width, selection_x, selection_y, selection_width, selection_height, _MUTED, 1)
    _centered_text(rgb, width, router_center, 190, "Selected:", _LABEL)
    _centered_text(rgb, width, router_center, 218, "2 experts", _LABEL)
    for expert in range(9):
        row, column = divmod(expert, 5)
        row_y = expert_y + row * 440
        x = (650 if row == 0 else 745) + column * (expert_width + expert_gap)
        box_top, box_bottom = row_y + 8, row_y + 430
        _rectangle(rgb, width, x - 5, box_top, 180, box_bottom - box_top, _EXPERT_BORDER, 1)
        _centered_text(rgb, width, x + 80, box_top + 8, f"Expert {expert + 1}", _LABEL)
        _render_pair(rgb, width, state, f"experts.{expert}.0.weight", f"experts.{expert}.0.bias", "", x, row_y + 2)
        _render_pair(rgb, width, state, f"experts.{expert}.2.weight", f"experts.{expert}.2.bias", "", x, row_y + 209)
        _arrow(rgb, width, x + 80, row_y + 219, x + 80, row_y + 247)
    # The return uses a straight reserved lane and terminates immediately below the Final output weight matrix.
    output_return_x, output_matrix_bottom = 1240, 153
    _arrow(rgb, width, output_return_x, expert_top, output_return_x, output_matrix_bottom)

    # Top-right legend uses unused canvas space, leaving the full lower-left lane for the execution contract.
    legend_x = 1400
    _rectangle(rgb, width, legend_x, 20, 30, 30, _WEIGHT_BORDER, 2)
    _text(rgb, width, legend_x + 46, 25, "Weights", _LABEL)
    _rectangle(rgb, width, legend_x, 55, 30, 30, _BIAS_BORDER, 2)
    _text(rgb, width, legend_x + 46, 60, "Bias", _LABEL)
    for y, color, label in (
        (90, _color(1.0), "large positive"),
        (117, _color(0.5), "moderate positive"),
        (144, (11, 16, 24), "neutral / zero"),
        (171, _color(-0.5), "negative"),
        (198, _color(-1.0), "large negative"),
    ):
        _fill_rectangle(rgb, width, legend_x, y, 20, 20, color)
        _rectangle(rgb, width, legend_x, y, 20, 20, _MUTED, 1)
        _text(rgb, width, legend_x + 34, y + 1, label, _LABEL)

    metadata = {
        "format": "crystal-9-tensor-inspector-v23", "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "tensor_count": len(state),
        "representation": "decoded inspector; not reconstructable",
        "proposed_deployment_tag": "lewismoten/crystal-9:q4",
        "execution_contract": {
            "public_input": "a-i; maximum 8 moves",
            "sequence": "BOS + history; PAD to 9 positions",
            "attention": "causal mask; read final non-PAD state",
            "routing": "softmax router; top 2 of 9 experts",
            "expert": "32 -> 32 SiLU -> 32",
            "public_output": "a-i; ! is invalid/no-move sentinel",
        },
        "normalization": "per-tensor symmetric max-absolute", "layout": "architecture-flow-v23",
        "legend_location": "top-right", "execution_contract_panel": {"location": "bottom-left", "bounds": [20, 750, 570, 470]},
        "attention_input_projection": {
            "group_label": "Attention input projections", "packed_weight_shape": [96, 32],
            "segments": {
                "Query": {"weight_shape": [32, 32], "bias_shape": [32]},
                "Key": {"weight_shape": [32, 32], "bias_shape": [32]},
                "Value": {"weight_shape": [32, 32], "bias_shape": [32]},
            },
        },
        "embedding_combination": {
            "operation": "elementwise addition",
            "inputs": ["token embedding", "position embedding"],
            "output": "attention input",
        },
        "bias_alignment": "vertical output-row axis", "sections": ["inputs", "attention", "norm_router", "experts", "output"],
        "legend": {"WEIGHTS": "matrix; rows are output features", "BIAS": "bias column; one value per output row", "B": "bias column; one value per output row"},
        "expert_layout": "five Expert boxes over four Expert boxes; each contains its first and second layer",
        "calculation_flow": ["embeddings and positions", "attention", "norm and router", "top-2 routed experts", "combined output"],
        "vocabulary": _VOCABULARY, "font": "DejaVu Sans", "router_selection_label": "Selected: 2 experts",
        "router_selection_lines": ["Selected:", "2 experts"], "router_selection_shape": "gray outlined rectangle",
        "router_path": "continuous downward arrow touching the Experts outline",
        "expert_return_path": "straight upward arrow ends below Final output matrix",
        "intra_expert_arrows": "compact clear gap between first and second layer matrices",
        "border_legend": {"dim purple": "weight matrix", "dim cyan": "bias vector"},
        "value_legend": {"yellow": "large positive", "green": "moderate positive", "black": "neutral / zero", "blue": "negative", "bright blue": "large negative"},
        "experts_outline": "slate gray",
        "tensors": inventory, "width": width, "height": height,
    }
    rows = b"".join(b"\0" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    png = b"".join((_SIGNATURE, _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                    _chunk(b"tEXt", b"crystal9-inspector\0" + str(metadata).encode()),
                    _chunk(b"IDAT", zlib.compress(rows, level=9)), _chunk(b"IEND", b"")))
    return png, metadata
