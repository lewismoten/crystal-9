"""Browser-safe binary projection of a validated Crystal-9 packed INT4 artifact.

This is not a new quantization.  It preserves the accepted artifact's packed
nibbles, scale bytes, tensor order/shape metadata, and source/integrity digests
in a format that a browser can parse without deserializing PyTorch pickle/ZIP.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch

from packed_int4 import FORMAT, PackedInt4Policy

BROWSER_FORMAT = "crystal-9-browser-packed-int4-v1"
_MAGIC = b"C9B1"
_MAX_HEADER_BYTES = 128_000


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tokenizer_metadata() -> dict[str, object]:
    design = json.loads((Path(__file__).resolve().parents[1] / "design.json").read_text())
    return {
        "tokens": design["vocabulary"]["tokens"],
        "input_symbols": design["vocabulary"]["input_symbols"],
        "max_history_moves": design["architecture"]["context_window"],
        "invalid_output": design["vocabulary"]["invalid_output"],
    }


def export_browser_artifact(source_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """Write a browser-native view of an already validated packed INT4 artifact."""
    source_path = Path(source_path)
    raw_source = source_path.read_bytes()
    manifest = torch.load(source_path, map_location="cpu", weights_only=True)
    # Reuse the deployment runtime's strict format and integrity validation.
    PackedInt4Policy(manifest)
    if manifest["format"] != FORMAT:
        raise ValueError("browser export currently requires float32-scale packed INT4 v1")

    payload = bytearray()
    tensors: dict[str, dict[str, object]] = {}
    for name in sorted(manifest["tensors"]):
        record = manifest["tensors"][name]
        scales = record["scales"].detach().cpu().contiguous().numpy().tobytes()
        packed = record["packed"].detach().cpu().contiguous().numpy().tobytes()
        scale_offset = len(payload)
        payload.extend(scales)
        packed_offset = len(payload)
        payload.extend(packed)
        tensors[name] = {
            "shape": list(record["shape"]),
            "scheme": record["scheme"],
            "group_size": record["group_size"],
            "count": record["count"],
            "scale_dtype": "float32-le",
            "scales": {"offset": scale_offset, "length": len(scales)},
            "packed": {"offset": packed_offset, "length": len(packed)},
        }

    header = {
        "format": BROWSER_FORMAT,
        "source_artifact_sha256": hashlib.sha256(raw_source).hexdigest(),
        "integrity_sha256": manifest["integrity_sha256"],
        "architecture": manifest["architecture"],
        "norm_weight_group_size": manifest["norm_weight_group_size"],
        "parameter_values": manifest["parameter_values"],
        "tokenizer": _tokenizer_metadata(),
        "tensors": tensors,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    header_bytes = _json(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError("browser artifact header exceeds bound")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + payload)
    return {key: value for key, value in header.items() if key != "tensors"}


def decode_browser_artifact(blob: bytes) -> dict[str, object]:
    """Parse a browser artifact and return raw packed tensor records for tests."""
    if len(blob) < 8 or blob[:4] != _MAGIC:
        raise ValueError("not a Crystal-9 browser packed artifact")
    header_size = struct.unpack(">I", blob[4:8])[0]
    if not 2 <= header_size <= _MAX_HEADER_BYTES or 8 + header_size > len(blob):
        raise ValueError("invalid Crystal-9 browser header length")
    header = json.loads(blob[8 : 8 + header_size])
    if header.get("format") != BROWSER_FORMAT:
        raise ValueError("unsupported Crystal-9 browser format")
    payload = blob[8 + header_size :]
    if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
        raise ValueError("browser packed payload integrity validation failed")
    tensors: dict[str, dict[str, object]] = {}
    for name, record in header["tensors"].items():
        scale = record["scales"]
        packed = record["packed"]
        scale_end = int(scale["offset"]) + int(scale["length"])
        packed_end = int(packed["offset"]) + int(packed["length"])
        if min(int(scale["offset"]), int(packed["offset"])) < 0 or max(scale_end, packed_end) > len(payload):
            raise ValueError(f"tensor {name} references bytes outside payload")
        tensors[name] = {
            **record,
            "scales": payload[int(scale["offset"]) : scale_end],
            "packed": payload[int(packed["offset"]) : packed_end],
        }
    return {**header, "tensors": tensors}
