import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "int2_fp8_scale_expert_order_search_hierarchy_screen.py"
    spec = importlib.util.spec_from_file_location("expert_order_search_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ordered_expert_interleave_round_trips_nontrivial_expert_order():
    module = _module()
    streams = (b"abcdef", b"ABCDEF", b"123456")

    encoded = module.encode_ordered_expert_interleave(streams, order=(2, 0, 1))

    assert encoded == b"1aA2bB3cC4dD5eE6fF"
    assert module.decode_ordered_expert_interleave(encoded, expert_count=3, stream_length=6, order=(2, 0, 1)) == streams


def test_screen_includes_canonical_order_as_a_deterministic_baseline():
    module = _module()

    report = module.screen(orders=((0, 1, 2, 3, 4, 5, 6, 7, 8),))

    assert report["screen"]["orders_screened"] == 1
    assert report["screen"]["best_order"] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
