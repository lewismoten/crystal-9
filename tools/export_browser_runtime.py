#!/usr/bin/env python3
"""Build the browser-executable PNG transport from the accepted packed INT4 artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.browser_packed_int4 import export_browser_artifact
from tools.model_png import decode_rgba_png, encode_rgba_png

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "crystal-9-int4-group2-packed-v1.pt"
RUNTIME = ROOT / "browser-runtime"
ARTIFACT = RUNTIME / "artifacts" / "crystal-9-int4-browser-v1.c9b"
PNG = ROOT / "png-model-gallery" / "models" / "int4-packed-browser-runtime.png"
MANIFEST = RUNTIME / "manifest.json"


def main() -> None:
    metadata = export_browser_artifact(SOURCE, ARTIFACT)
    payload = ARTIFACT.read_bytes()
    png = encode_rgba_png(payload, model_name="Crystal-9 browser packed INT4", precision="INT4 packed browser")
    restored, png_metadata = decode_rgba_png(png)
    if restored != payload:
        raise RuntimeError("browser artifact PNG round trip failed")
    PNG.parent.mkdir(parents=True, exist_ok=True)
    PNG.write_bytes(png)
    MANIFEST.write_text(json.dumps({
        "format": "crystal-9-browser-png-runtime-v1",
        "png": "/models/int4-packed-browser-runtime.png",
        "transport_bytes": len(payload),
        "transport_sha256": hashlib.sha256(payload).hexdigest(),
        "source_artifact_sha256": metadata["source_artifact_sha256"],
        "packed_integrity_sha256": metadata["integrity_sha256"],
        "parent_format": "crystal-9-packed-int4-v1",
        "png_format": png_metadata["format"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
