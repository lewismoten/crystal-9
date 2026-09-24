"""Exhaustive acceptance check for the packed FP8-scale INT2 research candidate."""

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
OUTPUT = ROOT / "artifacts/int2-packed-scalar-fp8-e4m3fn-scales-preflight-20260924"


def main() -> None:
    OUTPUT.mkdir(exist_ok=False)
    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(tokenizer.vocab_size).eval()
    source.load_state_dict(checkpoint["state_dict"])
    artifact = OUTPUT / "crystal-9-int2-packed-scalar-fp8-e4m3fn-scales.pt"
    manifest = export_packed_int2(source, artifact, scale_dtype=torch.float8_e4m3fn)
    histories = [history for history in legal_histories() if optimal_move(history) != "!"]
    acceptance = evaluate_packed(PackedInt2Policy.load(artifact).eval(), tokenizer, torch.device("cpu"), histories, optimal_move)
    report = {
        "layout": "complete-scalar-group-int2-packed-fp8-e4m3fn-scales",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "format": manifest["format"],
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "integrity_sha256": manifest["integrity_sha256"],
        "quantization": {
            "bits": 2,
            "packing": "2-bit scalar codes packed low-bit-first",
            "scale_type": "float8_e4m3fn",
            "scale_count": manifest["parameter_values"],
            "storage_efficient": False,
            "representation": "hierarchy-quantization research candidate; scalar scales compressed to float8_e4m3fn",
        },
        "acceptance": acceptance,
        "decision": "accepted" if acceptance == {"legal_histories": 294778, "policy_misses": 0} else "rejected",
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["decision"] != "accepted":
        raise SystemExit("Packed FP8-scale INT2 exhaustive policy gate failed")


if __name__ == "__main__":
    main()
