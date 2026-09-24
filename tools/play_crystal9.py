#!/usr/bin/env python3
"""Play a text-only game of tic-tac-toe against a local Crystal-9 artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crystal9 import GameTokenizer
from packed_int4 import PackedInt4Policy

SQUARES = "abcdefghi"
WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 4, 6))
DEFAULT_ARTIFACT = ROOT / "artifacts/crystal-9-int4-group2-packed-fp16-scales-v1.pt"


def _grid(cells: list[str]) -> str:
    border = "+---+---+---+"
    rows = [border]
    for start in range(0, 9, 3):
        rows.append("|" + "|".join(f" {cell} " for cell in cells[start : start + 3]) + "|")
        rows.append(border)
    return "\n".join(rows)


def available_moves_text(history: str) -> str:
    """Show only currently legal a-i input labels; occupied squares are blank."""
    return _grid([symbol if symbol not in history else " " for symbol in SQUARES])


def board_text(history: str) -> str:
    """Show the board state alone, with empty squares left blank."""
    cells = [" "] * 9
    for turn, move in enumerate(history):
        cells[SQUARES.index(move)] = "X" if turn % 2 == 0 else "O"
    return _grid(cells)


def game_view(history: str) -> str:
    """Render the actual board on the left and legal moves on the right."""
    board_lines = board_text(history).splitlines()
    move_lines = available_moves_text(history).splitlines()
    gap = "        "
    left_width = len(board_lines[0])
    header = f"{'Board':<{left_width}}{gap}Available moves"
    return "\n".join([header] + [f"{board}{gap}{moves}" for board, moves in zip(board_lines, move_lines)])


def winner(history: str) -> str | None:
    cells = list(SQUARES)
    for turn, move in enumerate(history):
        cells[SQUARES.index(move)] = "X" if turn % 2 == 0 else "O"
    for row in WINS:
        marks = {cells[index] for index in row}
        if marks == {"X"}:
            return "X"
        if marks == {"O"}:
            return "O"
    return None


def game_status(history: str) -> str | None:
    won = winner(history)
    if won:
        return f"{won} wins"
    if len(history) == 9:
        return "Draw"
    return None


def apply_move(history: str, move: str) -> str:
    move = move.lower().strip()
    if move not in SQUARES or len(move) != 1:
        raise ValueError("choose one square from a through i")
    if move in history:
        raise ValueError(f"square {move} is already occupied")
    if game_status(history):
        raise ValueError("the game is already over")
    return history + move


def choose_player(value: str | None) -> str:
    if value:
        candidate = value.upper()
        if candidate in {"X", "O"}:
            return candidate
        raise ValueError("player must be X or O")
    while True:
        answer = input("Play as X (first) or O (second)? ").strip().upper()
        if answer in {"X", "O"}:
            return answer
        print("Enter X or O.")


def model_move(runtime: PackedInt4Policy, tokenizer: GameTokenizer, history: str) -> str:
    move = runtime.predict(history, tokenizer)
    if move not in SQUARES or move in history:
        raise RuntimeError(f"model returned unusable move {move!r} for history {history!r}")
    return move


def play(runtime: PackedInt4Policy, tokenizer: GameTokenizer, human: str) -> None:
    history = ""
    model_player = "O" if human == "X" else "X"
    print(f"You are {human}; Crystal-9 is {model_player}. Enter a-i, or q to quit.")

    while not game_status(history):
        print("\n" + game_view(history))
        turn = "X" if len(history) % 2 == 0 else "O"
        if turn == human:
            while True:
                move = input(f"Your move ({human}): ").strip().lower()
                if move in {"q", "quit", "exit"}:
                    print("Game ended.")
                    return
                try:
                    history = apply_move(history, move)
                    break
                except ValueError as error:
                    print(error)
        else:
            move = model_move(runtime, tokenizer, history)
            history = apply_move(history, move)
            print(f"Crystal-9 ({model_player}) chooses {move}.")

    print("\n" + board_text(history))
    print(game_status(history))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=("X", "O", "x", "o"), help="play first as X or second as O")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="packed Crystal-9 INT4 artifact")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        raise SystemExit(f"artifact not found: {artifact}")
    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    runtime = PackedInt4Policy.load(artifact).eval()
    print(f"Loaded {artifact.name} ({runtime.manifest['format']}).")
    play(runtime, tokenizer, choose_player(args.player))


if __name__ == "__main__":
    main()
