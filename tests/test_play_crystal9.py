import pytest

from tools.play_crystal9 import apply_move, available_moves_text, board_text, game_status, game_view


def test_available_moves_and_playfield_are_separate_after_moves():
    available = available_moves_text("aei")
    board = board_text("aei")

    assert " a " not in available
    assert " e " not in available
    assert " i " not in available
    assert " b " in available
    assert " X " in board
    assert " O " in board
    assert " a " not in board
    assert "+---+---+---+" in available
    assert "+---+---+---+" in board


def test_game_view_places_board_left_and_available_moves_right():
    lines = game_view("aei").splitlines()

    assert lines[0].startswith("Board")
    assert lines[0].endswith("Available moves")
    assert "| X |" in lines[2]
    assert lines[2].endswith("|   | b | c |")
    assert "| O |" in lines[4]


def test_apply_move_rejects_nonempty_or_invalid_square():
    assert apply_move("ae", "b") == "aeb"

    with pytest.raises(ValueError, match="already occupied"):
        apply_move("ae", "a")
    with pytest.raises(ValueError, match="a through i"):
        apply_move("ae", "z")


def test_game_status_identifies_winner_and_draw():
    assert game_status("adbec") == "X wins"
    assert game_status("aecgibhfd") == "Draw"
    assert game_status("ae") is None
