import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_bitfield_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("bitfield_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bitfield_stream_round_trips_exact_fp8_bytes():
    module = _module()
    scale_bytes = bytes([0x00, 0x39, 0xB7, 0x7F, 0x41])

    upper, mantissas = module.encode_fp8_bitfields(scale_bytes)

    assert upper == bytes([0x00, 0x07, 0x16, 0x0F, 0x08])
    assert module.decode_fp8_bitfields(upper, mantissas) == scale_bytes


def test_bitfield_stream_rejects_mismatched_component_lengths():
    module = _module()

    try:
        module.decode_fp8_bitfields(b"\x01", b"")
    except ValueError as error:
        assert "length" in str(error)
    else:
        raise AssertionError("mismatched components must be rejected")
