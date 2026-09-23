from training import optimal_move


def test_minimax_oracle_selects_center_after_a_corner_opening():
    assert optimal_move("a") == "e"


def test_minimax_oracle_marks_repeated_moves_invalid():
    assert optimal_move("aa") == "!"
