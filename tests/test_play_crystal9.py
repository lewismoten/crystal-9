import subprocess
import sys
from pathlib import Path

from tools.play_crystal9 import board, status

ROOT = Path(__file__).resolve().parents[1]


def test_board_uses_three_plain_dot_o_x_rows():
    assert board("aei") == "x..\n.o.\n..x"


def test_status_handles_rows_columns_diagonals_and_draws():
    assert status("adbec") == "X wins"
    assert status("abeci") == "X wins"
    assert status("aec") is None
    assert status("aecgibhfd") == "Draw"


def test_demo_reports_invalid_move_without_explanation():
    result = subprocess.run(
        [sys.executable, "tools/play_crystal9.py"],
        cwd=ROOT,
        input="z\na\nb\nc\nd\n",
        capture_output=True,
        text=True,
        check=True,
    )

    assert "invalid move" in result.stdout
    assert "Choose an empty square" not in result.stdout
