#!/usr/bin/env python3
"""Generate the deterministic Crystal-9 / Palace-9 tic-tac-toe corpus.

The script uses only Python's standard library. It intentionally emits move
histories and next-move labels, not winner labels or winning-line annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from functools import lru_cache
from itertools import product
from pathlib import Path

SQUARES = "abcdefghi"
MOVE_ORDER = "ebdfhcgia"
WINS = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def winner(board: str) -> str | None:
    """Return the side that occupies a three-in-a-row, if any."""
    for first, second, third in WINS:
        if board[first] != "." and board[first] == board[second] == board[third]:
            return board[first]
    return None


def board_for(history: str) -> str | None:
    """Return an X/O/. board for a legal chronology, otherwise None.

    A move after a terminal board is invalid. No caller receives a winning-line
    label: terminality only prevents a continuation from being legal.
    """
    board = ["."] * 9
    for index, symbol in enumerate(history):
        if symbol not in SQUARES:
            return None
        square = SQUARES.index(symbol)
        if board[square] != "." or winner("".join(board)):
            return None
        board[square] = "X" if index % 2 == 0 else "O"
    return "".join(board)


@lru_cache(maxsize=None)
def score(board: str, turn: str) -> int:
    """Minimax value from X's perspective: win=1, draw=0, loss=-1."""
    won = winner(board)
    if won == "X":
        return 1
    if won == "O":
        return -1
    if "." not in board:
        return 0
    values = [
        score(board[:square] + turn + board[square + 1 :], "O" if turn == "X" else "X")
        for square in range(9)
        if board[square] == "."
    ]
    return max(values) if turn == "X" else min(values)


def optimal_moves(history: str) -> list[str]:
    """All minimax-optimal continuations in declared deterministic tie order."""
    board = board_for(history)
    if board is None or winner(board) or "." not in board:
        return []
    turn = "X" if len(history) % 2 == 0 else "O"
    candidates: list[tuple[int, str]] = []
    for symbol in MOVE_ORDER:
        square = SQUARES.index(symbol)
        if board[square] == ".":
            candidate = board[:square] + turn + board[square + 1 :]
            candidates.append((score(candidate, "O" if turn == "X" else "X"), symbol))
    best = (max if turn == "X" else min)(value for value, _ in candidates)
    return [symbol for value, symbol in candidates if value == best]


def optimal_move(history: str) -> str:
    """One deterministic target; `!` is reserved for invalid/no-continuation input."""
    choices = optimal_moves(history)
    return choices[0] if choices else "!"


def legal_histories(prefix: str = "", maximum_length: int = 8) -> list[str]:
    """All legal non-post-terminal histories through the model's 8-move input limit."""
    board = board_for(prefix)
    if board is None or winner(board) or len(prefix) == maximum_length:
        return [prefix]
    histories = [prefix]
    for symbol in SQUARES:
        if board[SQUARES.index(symbol)] == ".":
            histories.extend(legal_histories(prefix + symbol, maximum_length))
    return histories


def reachable_boards() -> set[str]:
    """Every board reachable by alternating legal play, including terminal boards."""
    return {board for history in legal_histories(maximum_length=9) if (board := board_for(history)) is not None}


def board_text(board: str) -> str:
    return "\n".join(" ".join(board[offset : offset + 3]) for offset in range(0, 9, 3))


def board_grid(board: str) -> str:
    return "/".join(board[offset : offset + 3] for offset in range(0, 9, 3))


def _transform_index(index: int, rotations: int, reflect: bool) -> int:
    row, column = divmod(index, 3)
    for _ in range(rotations):
        row, column = column, 2 - row
    if reflect:
        column = 2 - column
    return row * 3 + column


TRANSFORMS = tuple(
    tuple(_transform_index(index, rotations, reflect) for index in range(9))
    for rotations in range(4)
    for reflect in (False, True)
)
ROTATIONS = TRANSFORMS[::2]


def transform_board(board: str, mapping: tuple[int, ...]) -> str:
    transformed = ["."] * 9
    for old, new in enumerate(mapping):
        transformed[new] = board[old]
    return "".join(transformed)


def canonical_board(board: str, transforms: tuple[tuple[int, ...], ...]) -> str:
    return min(transform_board(board, mapping) for mapping in transforms)


def compute_statistics() -> dict[str, int]:
    """Compute raw, reachable, and canonical board-space counts."""
    all_boards = ["".join(board) for board in product(".XO", repeat=9)]
    reachable = reachable_boards()
    nonterminal_histories = [history for history in legal_histories() if optimal_move(history) != "!"]
    return {
        "all_board_states": len(all_boards),
        "reachable_board_states": len(reachable),
        "invalid_board_states": len(all_boards) - len(reachable),
        "all_board_states_rotation_canonical": len({canonical_board(board, ROTATIONS) for board in all_boards}),
        "all_board_states_rotation_and_reflection_canonical": len({canonical_board(board, TRANSFORMS) for board in all_boards}),
        "reachable_board_states_rotation_canonical": len({canonical_board(board, ROTATIONS) for board in reachable}),
        "reachable_board_states_rotation_and_reflection_canonical": len({canonical_board(board, TRANSFORMS) for board in reachable}),
        "nonterminal_legal_histories": len(nonterminal_histories),
    }


def json_lines(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_rows(histories: list[str]) -> Iterable[dict[str, object]]:
    for history in histories:
        board = board_for(history)
        assert board is not None
        choices = optimal_moves(history)
        if not choices:
            continue
        yield {
            "history": history,
            "next_player": "X" if len(history) % 2 == 0 else "O",
            "board": board_grid(board),
            "board_text": board_text(board),
            "optimal_move": choices[0],
            "optimal_moves": choices,
        }


def board_rows(boards: Iterable[str]) -> Iterable[dict[str, object]]:
    for board in sorted(boards):
        yield {"board": board_grid(board), "board_text": board_text(board)}


def generate(output: Path) -> dict[str, object]:
    """Write all deterministic corpus files and return their manifest mapping."""
    output.mkdir(parents=True, exist_ok=True)
    histories = [history for history in legal_histories() if optimal_move(history) != "!"]
    reachable = reachable_boards()
    nonterminal_boards = {board_for(history) for history in histories}
    assert None not in nonterminal_boards
    rotation_canonical = {canonical_board(board, ROTATIONS) for board in reachable}
    dihedral_canonical = {canonical_board(board, TRANSFORMS) for board in reachable}

    files: dict[str, dict[str, object]] = {}
    artifacts: list[tuple[str, Iterable[dict[str, object]]]] = [
        ("nonterminal-legal-histories.jsonl", history_rows(histories)),
        ("reachable-board-states.jsonl", board_rows(reachable)),
        ("nonterminal-board-states.jsonl", board_rows(nonterminal_boards)),
        ("rotation-canonical-reachable-board-states.jsonl", board_rows(rotation_canonical)),
        ("dihedral-canonical-reachable-board-states.jsonl", board_rows(dihedral_canonical)),
    ]
    for name, rows in artifacts:
        path = output / name
        files[name] = {"rows": json_lines(path, rows), "sha256": digest(path)}

    manifest: dict[str, object] = {
        "format_version": 1,
        "generator": "generate_dataset.py",
        "tie_order": MOVE_ORDER,
        "counts": compute_statistics(),
        "files": files,
        "contract": {
            "history": "chronological occupied-square letters",
            "optimal_move": "first minimax-optimal move in tie_order",
            "optimal_moves": "all minimax-optimal moves in tie_order",
            "terminal_semantics": "No win or winning-line labels are emitted; a continuation after terminal play is invalid.",
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    manifest = generate(args.output)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
