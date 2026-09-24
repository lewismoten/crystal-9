from tools.play_crystal9 import board, status


def test_board_uses_three_plain_dot_o_x_rows():
    assert board("aei") == "x..\n.o.\n..x"


def test_status_handles_rows_columns_diagonals_and_draws():
    assert status("adbec") == "X wins"
    assert status("abeci") == "X wins"
    assert status("aec") is None
    assert status("aecgibhfd") == "Draw"
