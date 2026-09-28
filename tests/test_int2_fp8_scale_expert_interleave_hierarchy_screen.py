import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_expert_interleave_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("expert_interleave_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expert_interleave_round_trips_corresponding_scale_streams():
    module = _module()
    streams = (b"abc", b"DEF", b"123")

    encoded = module.encode_expert_interleave(streams)

    assert encoded == b"aD1bE2cF3"
    assert module.decode_expert_interleave(encoded, expert_count=3, stream_length=3) == streams


def test_expert_interleave_rejects_unequal_stream_lengths():
    module = _module()

    try:
        module.encode_expert_interleave((b"ab", b"c"))
    except ValueError as error:
        assert "equal" in str(error)
    else:
        raise AssertionError("unequal expert streams must be rejected")
