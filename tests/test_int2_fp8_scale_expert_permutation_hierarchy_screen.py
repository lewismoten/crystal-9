import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_expert_permutation_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("expert_permutation_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_permute_expert_streams_round_trips_a_fixed_order():
    module = _module()
    streams = (b"abc", b"DEF", b"123")

    ordered = module.permute_expert_streams(streams, (2, 0, 1))

    assert ordered == (b"123", b"abc", b"DEF")
    assert module.restore_expert_streams(ordered, (2, 0, 1)) == streams


def test_permute_expert_streams_rejects_non_permutations():
    module = _module()

    try:
        module.permute_expert_streams((b"a", b"b"), (0, 0))
    except ValueError as error:
        assert "permutation" in str(error)
    else:
        raise AssertionError("invalid expert order must be rejected")
