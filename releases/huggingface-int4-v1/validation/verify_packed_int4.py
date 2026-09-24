"""Verify the staged packed-INT4 artifact against the legal-history oracle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
sys.path.insert(1, str(ROOT))

from crystal9 import GameTokenizer  # noqa: E402
from packed_int4 import PackedInt4Policy, evaluate_packed  # noqa: E402
from training import legal_histories, optimal_move  # noqa: E402

artifact = STAGE / "artifacts/crystal-9-int4-group2-packed-v1.pt"
runtime = PackedInt4Policy.load(artifact).eval()
tokenizer = GameTokenizer.from_design_file(STAGE / "design.json")
histories = [history for history in legal_histories() if optimal_move(history) != "!"]
acceptance = evaluate_packed(runtime, tokenizer, torch.device("cpu"), histories, optimal_move)
report = {
    "artifact": artifact.name,
    "format": runtime.manifest["format"],
    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    "integrity_sha256": runtime.manifest["integrity_sha256"],
    "architecture": runtime.architecture,
    "runtime": "PackedInt4Policy",
    "acceptance": acceptance,
    "invalid_input_examples": {history: runtime.predict(history, tokenizer) for history in ["!", "aa", "abcdefghi", "adbecf"]},
}
print(json.dumps(report, indent=2))
if acceptance != {"legal_histories": 294778, "policy_misses": 0}:
    raise SystemExit("Packed INT4 exhaustive policy gate failed")
if any(result != "!" for result in report["invalid_input_examples"].values()):
    raise SystemExit("Packed INT4 invalid-history gate failed")
(STAGE / "validation/packed-int4-acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
