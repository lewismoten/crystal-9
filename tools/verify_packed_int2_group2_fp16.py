"""Exhaustive preflight for the two-value shared-FP16-scale INT2 candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crystal9 import GameTokenizer, TinyMoEPolicy
from packed_int2 import evaluate_packed
from packed_int2_group2_fp16_scales import PackedInt2Group2FP16ScalePolicy, export_packed_int2_group2_fp16_scales
from training import legal_histories, optimal_move

SOURCE = ROOT / "artifacts-fp32.pt"
OUTPUT = ROOT / "artifacts/rejected/int2-packed-group2-fp16-scales-preflight-20260924"
LAYOUT = "complete-int2-two-value-shared-fp16-scales-research"


def run_preflight(source_path: Path, output_dir: Path, histories: list[str], device: torch.device) -> dict:
    """Export and evaluate a distinct two-value shared-FP16-scale candidate."""
    output_dir.mkdir(exist_ok=False)
    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(tokenizer.vocab_size).eval()
    source.load_state_dict(checkpoint["state_dict"])
    artifact = output_dir / "crystal-9-int2-packed-group2-fp16-scales.pt"
    manifest = export_packed_int2_group2_fp16_scales(source, artifact)
    acceptance = evaluate_packed(PackedInt2Group2FP16ScalePolicy.load(artifact).eval(), tokenizer, device, histories, optimal_move)
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
            "packing": "2-bit codes packed low-bit-first",
            "group_size": manifest["group_size"],
            "scale_type": manifest["scale_type"],
            "scale_count": manifest["scale_count"],
            "storage_efficient": True,
            "representation": "distinct two-value shared-scale hierarchy-quantization research candidate",
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
        raise SystemExit("Packed group-2 FP16-scale INT2 exhaustive policy gate failed")


if __name__ == "__main__":
    main()
