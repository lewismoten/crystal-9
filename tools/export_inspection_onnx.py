"""Export Crystal-9's immutable F32 forward graph for inspection only."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training
from crystal9 import GameTokenizer


def export_f32_inspection_onnx(source_path: Path, output_path: Path) -> None:
    """Write a fixed `[1, 8]` F32 ONNX graph suitable for Netron inspection."""
    root = source_path.resolve().parent
    tokenizer = GameTokenizer.from_design_file(root / "design.json")
    model = training.load_reference_model(source_path, tokenizer.vocab_size, torch.device("cpu")).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (example,),
            str(output_path),
            input_names=["token_ids"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    export_f32_inspection_onnx(
        root / "artifacts-fp32.pt",
        root / "releases/huggingface-int4-v1/inspection/crystal-9-f32-fixed8-inspection.onnx",
    )
