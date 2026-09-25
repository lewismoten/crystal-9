import importlib.util
import subprocess
import sys
from pathlib import Path


def test_cross_expert_modular_residuals_round_trip_exact_scale_bytes():
    module_path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_cross_expert_modular_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("cross_expert_modular_screen", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    root = bytes([3, 250, 99, 0])
    peer = bytes([5, 1, 98, 255])
    residual = module.encode_modular_residual(root, peer)

    assert residual == bytes([2, 7, 255, 255])
    assert module.decode_modular_residual(root, residual) == peer


def test_cross_expert_transform_replaces_only_nonroot_expert_streams():
    module_path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_cross_expert_modular_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("cross_expert_modular_screen", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    streams = {
        "experts.0.0.weight": bytes([3, 250]),
        "experts.1.0.weight": bytes([5, 1]),
        "other": bytes([7]),
    }
    transformed, residual_names = module.transform_cross_expert_streams(streams)

    assert residual_names == ("experts.1.0.weight",)
    assert transformed["experts.0.0.weight"] == streams["experts.0.0.weight"]
    assert transformed["experts.1.0.weight"] == bytes([2, 7])
    assert module.decode_modular_residual(streams["experts.0.0.weight"], transformed["experts.1.0.weight"]) == streams["experts.1.0.weight"]


def test_cross_expert_screen_runs_from_the_project_root():
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "tools/int2_fp8_scale_cross_expert_modular_hierarchy_screen.py"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
