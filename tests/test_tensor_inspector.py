import hashlib
from pathlib import Path

from tools.tensor_inspector import render_checkpoint_inspector


def test_checkpoint_inspector_renders_actual_tensor_inventory_as_png():
    source = Path("artifacts-fp32.pt")

    png, metadata = render_checkpoint_inspector(source)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["tensor_count"] == 48
    assert metadata["tensors"]["attention.in_proj_weight"]["shape"] == [96, 32]
    assert metadata["tensors"]["experts.0.0.weight"]["shape"] == [32, 32]
    assert metadata["normalization"] == "per-tensor symmetric max-absolute"
