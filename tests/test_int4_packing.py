import torch

from crystal9 import pack_signed_int4, unpack_signed_int4


def test_signed_int4_pack_round_trips_odd_count():
    values = torch.tensor([-8, -7, -1, 0, 1, 6, 7], dtype=torch.int8)
    packed = pack_signed_int4(values)
    assert packed.dtype is torch.uint8
    assert packed.numel() == 4
    torch.testing.assert_close(unpack_signed_int4(packed, values.numel()), values)
