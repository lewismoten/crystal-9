from pathlib import Path
import subprocess
import sys

import onnx

from tools.export_inspection_onnx import export_f32_inspection_onnx


def test_export_f32_inspection_onnx_has_fixed_eight_token_input(tmp_path: Path):
    output = tmp_path / "crystal-9-f32-fixed8-inspection.onnx"

    export_f32_inspection_onnx(Path("artifacts-fp32.pt"), output)

    model = onnx.load(output)
    onnx.checker.check_model(model)
    assert model.graph.input[0].name == "token_ids"
    assert [item.dim_value for item in model.graph.input[0].type.tensor_type.shape.dim] == [1, 8]
    assert model.graph.output[0].name == "logits"
    assert len(model.graph.node) > 20
    assert output.with_suffix(".onnx.data").exists()


def test_export_script_runs_from_repository_root():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/export_inspection_onnx.py"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "releases/huggingface-int4-v1/inspection/crystal-9-f32-fixed8-inspection.onnx").exists()
