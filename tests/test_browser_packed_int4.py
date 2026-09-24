import hashlib
import json
import subprocess
from pathlib import Path

import torch

from packed_int4 import PackedInt4Policy
from tools.browser_packed_int4 import (
    BROWSER_FORMAT,
    decode_browser_artifact,
    export_browser_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/crystal-9-int4-group2-packed-v1.pt"


def test_browser_artifact_preserves_accepted_packed_tensor_records(tmp_path: Path):
    output = tmp_path / "crystal-9-int4-browser-v1.c9b"

    metadata = export_browser_artifact(SOURCE, output)
    decoded = decode_browser_artifact(output.read_bytes())
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)

    assert decoded["format"] == BROWSER_FORMAT
    assert metadata["source_artifact_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert decoded["source_artifact_sha256"] == metadata["source_artifact_sha256"]
    assert decoded["integrity_sha256"] == source["integrity_sha256"]
    assert set(decoded["tensors"]) == set(source["tensors"])
    for name, expected in source["tensors"].items():
        actual = decoded["tensors"][name]
        assert actual["shape"] == list(expected["shape"])
        assert actual["scheme"] == expected["scheme"]
        assert actual["group_size"] == expected["group_size"]
        assert actual["count"] == expected["count"]
        assert actual["scales"] == expected["scales"].numpy().tobytes()
        assert actual["packed"] == expected["packed"].numpy().tobytes()


def test_node_runtime_executes_browser_artifact_with_expected_valid_and_invalid_probes(tmp_path: Path):
    artifact = tmp_path / "crystal-9-int4-browser-v1.c9b"
    export_browser_artifact(SOURCE, artifact)
    runtime = PackedInt4Policy.load(SOURCE).eval()

    probes = ["", "e", "ea", "eac", "abcdefghi", "ee", "xyz"]
    expected = []
    from crystal9 import GameTokenizer

    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    for history in probes:
        expected.append({"history": history, "output": runtime.predict(history, tokenizer)})
    probe_path = tmp_path / "probes.json"
    probe_path.write_text(json.dumps(expected))

    completed = subprocess.run(
        ["node", "browser-runtime/test-runtime.mjs", str(artifact), str(probe_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == expected


def test_node_runtime_rejects_browser_artifact_with_tampered_parent_digest(tmp_path: Path):
    artifact = tmp_path / "crystal-9-int4-browser-v1.c9b"
    expected = export_browser_artifact(SOURCE, artifact)
    payload = bytearray(artifact.read_bytes())
    header_size = int.from_bytes(payload[4:8], "big")
    header = json.loads(payload[8 : 8 + header_size])
    header["source_artifact_sha256"] = "0" * 64
    altered_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    assert len(altered_header) == header_size
    payload[8 : 8 + header_size] = altered_header
    tampered = tmp_path / "tampered.c9b"
    tampered.write_bytes(payload)
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps({
        "source_artifact_sha256": expected["source_artifact_sha256"],
        "packed_integrity_sha256": expected["integrity_sha256"],
    }))

    probes_path = tmp_path / "probes.json"
    probes_path.write_text("[]")
    completed = subprocess.run(
        ["node", "browser-runtime/test-runtime.mjs", str(tampered), str(probes_path), str(expected_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "parent artifact digest mismatch" in completed.stderr


def test_minimal_browser_page_loads_png_transport_and_exposes_history_control():
    page = (ROOT / "browser-runtime" / "index.html").read_text()
    assert 'id="history"' in page
    assert 'id="run"' in page
    assert 'id="output"' in page
    assert 'id="model-image"' in page
    assert "Derivative browser projection of the accepted packed INT4 artifact" in page
    assert "models/int4-packed-browser-runtime.png" in (ROOT / "browser-runtime" / "manifest.json").read_text()
    assert "crypto.subtle.digest" in (ROOT / "browser-runtime" / "runtime.mjs").read_text()
    app = (ROOT / "browser-runtime" / "app.mjs").read_text()
    assert "source_artifact_sha256" in app
    assert "packed_integrity_sha256" in app
    assert "getImageData" in app
