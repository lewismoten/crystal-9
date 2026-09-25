"""Exhaustive acceptance check for fixed-binary LZMA-scale/permuted-bitplane-zlib-code INT2."""

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crystal9 import GameTokenizer, TinyMoEPolicy
from packed_int2 import evaluate_packed
from packed_int2_fp8_scale_lzma_fixed_binary_permuted_bitplane_zlib_codes import (
    PackedInt2Fp8ScaleLzmaFixedBinaryPermutedBitplaneZlibCodesPolicy,
    export_packed_int2_fp8_scale_lzma_fixed_binary_permuted_bitplane_zlib_codes,
)
from training import legal_histories, optimal_move

SOURCE = ROOT / "artifacts-fp32.pt"
OUTPUT = ROOT / "artifacts/int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-binary-permuted-bitplane-zlib-codes-preflight-20260925"


def build_report(source_path: Path, output: Path, histories: list[str], expected_move, vocab_size: int) -> dict:
    output.mkdir(exist_ok=False)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source = TinyMoEPolicy(vocab_size).eval()
    source.load_state_dict(checkpoint["state_dict"])
    artifact = output / "crystal-9-int2-packed-scalar-fp8-e4m3fn-scales-lzma-fixed-binary-permuted-bitplane-zlib-codes.c9i2"
    manifest = export_packed_int2_fp8_scale_lzma_fixed_binary_permuted_bitplane_zlib_codes(source, artifact)
    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    acceptance = evaluate_packed(PackedInt2Fp8ScaleLzmaFixedBinaryPermutedBitplaneZlibCodesPolicy.load(artifact).eval(), tokenizer, torch.device("cpu"), histories, expected_move)
    report = {
        "layout": "complete-scalar-group-int2-packed-fp8-e4m3fn-scales-lzma-fixed-binary-permuted-bitplane-zlib-codes",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "integrity_sha256": manifest["integrity_sha256"],
        "quantization": {
            "bits": 2,
            "packing": "2-bit scalar codes packed low-bit-first in a fixed-architecture binary container",
            "container": "fixed-architecture-binary",
            "code_packing": "permutation-selected-bitplane-transposed",
            "code_permutation": manifest["code_permutation"],
            "code_compression": "zlib-9",
            "scale_type": "float8_e4m3fn-lzma",
            "scale_count": manifest["parameter_values"],
            "raw_code_bytes": manifest["raw_code_bytes"],
            "compressed_code_bytes": manifest["compressed_code_bytes"],
            "raw_scale_bytes": manifest["raw_scale_bytes"],
            "compressed_scale_bytes": manifest["compressed_scale_bytes"],
            "storage_efficient": False,
            "representation": "lossless fixed-architecture binary transport of scalar FP8 scales and INT2 codes; research candidate",
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
        raise SystemExit("Fixed-binary LZMA-scale/permuted-bitplane-zlib-code INT2 exhaustive policy gate failed")


if __name__ == "__main__":
    main()
