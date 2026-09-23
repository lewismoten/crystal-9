from training import batch_ranges


def test_batch_ranges_limits_a_large_training_set_to_requested_batch_size():
    assert list(batch_ranges(5, 2)) == [(0, 2), (2, 4), (4, 5)]
