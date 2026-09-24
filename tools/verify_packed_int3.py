"""Exhaustive acceptance check for the packed complete INT3 Crystal-9 candidate."""

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crystal9 import GameTokenizer, TinyMoEPolicy
from packed_int3 import PackedInt3Policy, evaluate_packed, export_packed_int3
from training import legal_histories, optimal_move

SOURCE = ROOT / "artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt"
OUTPUT = ROOT / "artifacts/int3-packed-v1-preflight-20260924-retry1"


def main() -> None:
    OUTPUT.mkdir(exist_ok=False)
    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(tokenizer.vocab_size).eval()
    source.load_state_dict(checkpoint["state_dict"])
    artifact = OUTPUT / "crystal-9-int3-packed-v1.pt"
    manifest = export_packed_int3(source, artifact)
    histories = [history for history in legal_histories() if optimal_move(history) != "!"]
    acceptance = evaluate_packed(PackedInt3Policy.load(artifact).eval(), tokenizer, torch.device("cpu"), histories, optimal_move)
    report = {
        "layout": "complete-mixed-int3-scope14",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "format": manifest["format"],
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "integrity_sha256": manifest["integrity_sha256"],
        "quantization": {"bits": 3, "packing": "3-bit codes packed low-bit-first", "scale_type": "FP32", "layout": "expert/output rowwise; router group4; Q rowwise; K/V group2; scalar groups elsewhere"},
        "acceptance": acceptance,
        "decision": "accepted" if acceptance == {"legal_histories": 294778, "policy_misses": 0} else "rejected",
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["decision"] != "accepted":
        raise SystemExit("Packed INT3 exhaustive policy gate failed")


if __name__ == "__main__":
    main()
