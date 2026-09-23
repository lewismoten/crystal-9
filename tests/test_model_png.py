import hashlib

from tools.model_png import decode_rgba_png, encode_rgba_png


def test_model_byte_payload_round_trips_through_lossless_rgba_png():
    payload = bytes(range(256)) + b"crystal-9\x00payload"

    png = encode_rgba_png(payload)
    restored, metadata = decode_rgba_png(png)

    assert restored == payload
    assert metadata["source_bytes"] == len(payload)
    assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["width"] * metadata["height"] * 3 >= len(payload)
    # PNG truecolor (type 2) has RGB channels only; it does not store alpha.
    assert png[25] == 2
