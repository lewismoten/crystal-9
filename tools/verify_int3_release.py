"""Validate the static provenance manifest for the staged packed INT3 release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED = {"legal_histories": 294778, "policy_misses": 0}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_release(stage: Path) -> dict:
    manifest_path = stage / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("release_status") != "staged-not-published":
        raise ValueError("release must be staged, not published")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("release must declare exactly one packed INT3 artifact")
    artifact = artifacts[0]
    path = stage / artifact["path"]
    if not path.is_file() or path.stat().st_size != artifact["bytes"]:
        raise ValueError("artifact size does not match manifest")
    if _sha256(path) != artifact["sha256"]:
        raise ValueError("artifact digest does not match manifest")
    if artifact.get("acceptance") != EXPECTED:
        raise ValueError("artifact lacks exact exhaustive acceptance")
    if not artifact.get("source_checkpoint") or not artifact.get("source_report"):
        raise ValueError("artifact lacks immutable source provenance")
    return {"release_status": manifest["release_status"], "artifacts": [artifact]}
