#!/usr/bin/env python3
"""Export selected Crystal-9 artifacts as exact RGB PNG byte payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.model_png import decode_rgba_png, encode_rgba_png

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "png-model-gallery" / "models"
MODELS = (
    ("fp32-reference", "Immutable FP32 reference", "FP32", ROOT / "artifacts-fp32.pt", "accepted reference"),
    (
        "int4-full-qat",
        "Full-parameter INT4 QAT checkpoint",
        "INT4 QAT",
        ROOT / "artifacts/mixed-int4-full-parameters-norm-weight-group2-seed20260934-300-lr5e-5/artifacts-qat-mixed-int4-full-parameters-norm-weight-group2.pt",
        "accepted QAT checkpoint; FP32 masters retained",
    ),
    (
        "int4-packed-runtime",
        "Packed INT4 deployment artifact",
        "INT4 packed",
        ROOT / "artifacts/crystal-9-int4-group2-packed-v1.pt",
        "accepted packed runtime artifact",
    ),
    (
        "int3-suffix-qat",
        "Scoped INT3 suffix QAT checkpoint",
        "INT3 suffix QAT",
        ROOT / "artifacts/int3-suffix-qat-300-continuation-seed20260936-lr5e-5/artifacts-qat-mixed-int3-suffix.pt",
        "accepted staged scope; upstream tensors remain F32",
    ),
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for slug, name, precision, source, status in MODELS:
        payload = source.read_bytes()
        png = encode_rgba_png(payload, model_name=name, precision=precision)
        restored, png_metadata = decode_rgba_png(png)
        if restored != payload:
            raise RuntimeError(f"PNG round trip failed: {source}")
        target = OUTPUT / f"{slug}.png"
        target.write_bytes(png)
        manifest.append(
            {
                "id": slug,
                "name": name,
                "status": status,
                "precision": precision,
                "source": source.relative_to(ROOT).as_posix(),
                "source_bytes": len(payload),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "png": f"models/{target.name}",
                "png_bytes": len(png),
                "width": png_metadata["width"],
                "height": png_metadata["height"],
                "transport": "RGB payload bytes; no alpha channel",
                "round_trip_verified": True,
            }
        )
    (OUTPUT.parent / "manifest.json").write_text(json.dumps({"format": "crystal-9-rgb-byte-png-v3", "models": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
