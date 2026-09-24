#!/usr/bin/env python3
"""Export tensor-aware inspection maps for trusted Crystal-9 checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.tensor_inspector import render_checkpoint_inspector

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "png-model-gallery" / "inspectors"
MODELS = (
    ("fp32-reference", "Immutable FP32 reference", ROOT / "artifacts-fp32.pt"),
    ("int4-full-qat", "Full-parameter INT4 QAT checkpoint", ROOT / "artifacts/mixed-int4-full-parameters-norm-weight-group2-seed20260934-300-lr5e-5/artifacts-qat-mixed-int4-full-parameters-norm-weight-group2.pt"),
    ("int3-suffix-qat", "Scoped INT3 suffix QAT checkpoint", ROOT / "artifacts/int3-suffix-qat-300-continuation-seed20260936-lr5e-5/artifacts-qat-mixed-int3-suffix.pt"),
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inspectors = []
    for slug, name, source in MODELS:
        png, metadata = render_checkpoint_inspector(source)
        target = OUTPUT / f"{slug}-tensors.png"
        target.write_bytes(png)
        inspectors.append({
            "id": slug, "name": name, "png": f"inspectors/{target.name}", "png_bytes": len(png),
            "source": source.relative_to(ROOT).as_posix(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "tensor_count": metadata["tensor_count"], "width": metadata["width"], "height": metadata["height"],
            "normalization": metadata["normalization"], "derived": True,
        })
    (OUTPUT.parent / "inspectors.json").write_text(json.dumps({"format": "crystal-9-tensor-inspector-v11", "inspectors": inspectors}, indent=2) + "\n")


if __name__ == "__main__":
    main()
