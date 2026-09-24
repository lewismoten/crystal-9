"""Exhaustive acceptance check for gzip-transported FP8-scale packed INT2."""

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crystal9 import GameTokenizer, TinyMoEPolicy
from packed_int2 import evaluate_packed
from packed_int2_fp8_scale_gzip import PackedInt2Fp8ScaleGzipPolicy, export_packed_int2_fp8_scale_gzip
from training import legal_histories, optimal_move

SOURCE = ROOT / "artifacts-fp32.pt"
OUTPUT = ROOT / "artifacts/int2-packed-scalar-fp8-e4m3fn-scales-gzip-preflight-20260924"


def build_report(source_path: Path, output: Path, histories: list[str], expected_move, vocab_size: int) -> dict:
    """Export, evaluate, and record one immutable gzip transport candidate."""
    output.mkdir(exist_ok=False)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size).eval()
    source.load_state_dict(checkpoint["state_dict"])
    artifact = output / "crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-gzip.pt"
    manifest = export_packed_int2_fp8_scale_gzip(source, artifact)
    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    acceptance = evaluate_packed(PackedInt2Fp8ScaleGzipPolicy.load(artifact).eval(), tokenizer, torch.device("cpu"), histories, expected_move)
    report = {
        "layout": "complete-scalar-group-int2-packed-fp8-e4m3fn-scales-gzip",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "integrity_sha256": manifest["integrity_sha256"],
        "quantization": {
            "bits": 2,
            "packing": "2-bit scalar codes packed low-bit-first",
            "scale_type": "float8_e4m3fn-gzip",
            "scale_count": manifest["parameter_values"],
            "raw_scale_bytes": manifest["raw_scale_bytes"],
            "compressed_scale_bytes": manifest["compressed_scale_bytes"],
            "storage_efficient": False,
            "representation": "lossless gzip transport of scalar FP8 scales; research candidate",
        },
        "acceptance": acceptance,
        "decision": "accepted" if acceptance == {"legal_histories": 294778, "policy_misses": 0} else "rejected",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    histories = [history for history in legal_histories() if optimal_move(history) != "!"]
    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    report = build_report(SOURCE, OUTPUT, histories, optimal_move, tokenizer.vocab_size)
    print(json.dumps(report, indent=2))
    if report["decision"] != "accepted":
        raise SystemExit("Packed gzip-scale INT2 exhaustive policy gate failed")


if __name__ == "__main__":
    main()
