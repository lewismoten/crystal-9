import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_row_baseline_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("row_baseline_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_row_mode_residual_stream_round_trips_each_row_exactly():
    module = _module()
    rows = (bytes([4, 4, 9, 4]), bytes([8, 1, 8, 8]))

    baselines, residual = module.encode_row_mode_residuals(rows)

    assert baselines == bytes([4, 8])
    assert module.decode_row_mode_residuals((len(row) for row in rows), baselines, residual) == rows


def test_row_mode_residual_rejects_invalid_baseline_inventory():
    module = _module()

    try:
        module.decode_row_mode_residuals((2,), b"", b"\x00\x00")
    except ValueError as error:
        assert "baseline" in str(error)
    else:
        raise AssertionError("missing baseline must be rejected")
