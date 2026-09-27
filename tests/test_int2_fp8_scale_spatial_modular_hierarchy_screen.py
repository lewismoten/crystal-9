import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_spatial_modular_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("spatial_modular_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_row_modular_residual_stream_round_trips_each_row_exactly():
    module = _module()
    rows = (bytes([4, 7, 2, 2]), bytes([250, 3, 255]))

    roots, residuals = module.encode_row_modular_residuals(rows)

    assert roots == bytes([4, 250])
    assert residuals == bytes([3, 251, 0, 9, 252])
    assert module.decode_row_modular_residuals((len(row) for row in rows), roots, residuals) == rows


def test_row_modular_residual_rejects_invalid_root_inventory():
    module = _module()

    try:
        module.decode_row_modular_residuals((2,), b"", b"\x01")
    except ValueError as error:
        assert "root" in str(error)
    else:
        raise AssertionError("missing row root must be rejected")
