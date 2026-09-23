from crystal9 import GameTokenizer


def test_custom_tokenizer_uses_the_declared_ids_and_prepends_bos():
    tokenizer = GameTokenizer.from_design_file("design.json")

    assert tokenizer.vocab_size == 13
    assert tokenizer.encode_history("ae") == [1, 4, 8]
    assert tokenizer.decode_id(12) == "i"


def test_custom_tokenizer_rejects_histories_longer_than_eight_moves():
    tokenizer = GameTokenizer.from_design_file("design.json")

    try:
        tokenizer.encode_history("abcdefghi")
    except ValueError as error:
        assert "at most 8" in str(error)
    else:
        raise AssertionError("a nine-move history must be rejected")
