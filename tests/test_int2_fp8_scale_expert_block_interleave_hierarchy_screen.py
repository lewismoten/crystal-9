import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_expert_block_interleave_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("expert_block_interleave_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_block_interleave_round_trips_corresponding_expert_streams():
    module = _module()
    streams = (b"abcdefg", b"ABCDEFG", b"1234567")

    encoded = module.encode_expert_block_interleave(streams, block_size=3)

    assert encoded == b"abcABC123defDEF456gG7"
    assert module.decode_expert_block_interleave(encoded, expert_count=3, stream_length=7, block_size=3) == streams


def test_block_interleave_rejects_invalid_payload_length():
    module = _module()

    try:
        module.decode_expert_block_interleave(b"abc", expert_count=2, stream_length=2, block_size=2)
    except ValueError as error:
        assert "length" in str(error)
    else:
        raise AssertionError("invalid block-interleave payload must be rejected")
