"""Exhaustive preflight for the FP8-E5M2 scalar-scale INT2 candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crystal9 import GameTokenizer, TinyMoEPolicy
from packed_int2 import PackedInt2Policy, evaluate_packed, export_packed_int2
from training import legal_histories, optimal_move

SOURCE = ROOT / "artifacts-fp32.pt"
OUTPUT = ROOT / "artifacts/rejected/int2-packed-scalar-fp8-e5m2-scales-preflight-20260924"
LAYOUT = "complete-scalar-group-int2-packed-fp8-e5m2-scales"


def run_preflight(source_path: Path, output_dir: Path, histories: list[str], device: torch.device) -> dict:
    """Export and evaluate a distinct E5M2 scale-encoding candidate."""
    output_dir.mkdir(exist_ok=False)
    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(tokenizer.vocab_size).eval()
    source.load_state_dict(checkpoint["state_dict"])
    artifact = output_dir / "crystal-9-int2-packed-scalar-fp8-e5m2-scales.pt"
    manifest = export_packed_int2(source, artifact, scale_dtype=torch.float8_e5m2)
    acceptance = evaluate_packed(PackedInt2Policy.load(artifact).eval(), tokenizer, device, histories, optimal_move)
    report = {
        "layout": LAYOUT,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "format": manifest["format"],
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "integrity_sha256": manifest["integrity_sha256"],
        "parameter_values": manifest["parameter_values"],
        "quantization": {
            "bits": 2,
            "packing": "2-bit scalar codes packed low-bit-first",
            "scale_type": "float8_e5m2",
            "scale_count": manifest["parameter_values"],
            "storage_efficient": False,
            "representation": "distinct scalar-scale hierarchy-quantization research candidate",
        },
        "acceptance": acceptance,
        "decision": "accepted" if acceptance == {"legal_histories": 294778, "policy_misses": 0} else "rejected",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    histories = [history for history in legal_histories() if optimal_move(history) != "!"]
    report = run_preflight(SOURCE, OUTPUT, histories, torch.device("cpu"))
    print(json.dumps(report, indent=2))
    if report["decision"] != "accepted":
        raise SystemExit("Packed FP8-E5M2-scale INT2 exhaustive policy gate failed")


if __name__ == "__main__":
    main()
