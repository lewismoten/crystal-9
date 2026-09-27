import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_row_column_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("row_column_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_row_column_stream_round_trips_exact_matrix_bytes():
    module = _module()
    rows = (bytes([4, 7, 2]), bytes([9, 1, 8]))

    column_stream = module.encode_row_column_stream(rows)

    assert column_stream == bytes([4, 9, 7, 1, 2, 8])
    assert module.decode_row_column_stream(column_stream, 2, 3) == rows


def test_row_column_stream_rejects_wrong_payload_length():
    module = _module()

    try:
        module.decode_row_column_stream(b"\x01", 2, 2)
    except ValueError as error:
        assert "length" in str(error)
    else:
        raise AssertionError("truncated column stream must be rejected")
