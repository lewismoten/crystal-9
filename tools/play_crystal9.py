#!/usr/bin/env python3
"""Play as X against the local Crystal-9 packed INT4 model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SQUARES = "abcdefghi"
WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 4, 6), (0, 4, 8), (2, 4, 6))
ARTIFACT = ROOT / "artifacts/crystal-9-int4-group2-packed-fp16-scales-v1.pt"


def board(history: str) -> str:
    cells = ["."] * 9
    for turn, move in enumerate(history):
        cells[SQUARES.index(move)] = "xo"[turn % 2]
    return "\n".join("".join(cells[row : row + 3]) for row in range(0, 9, 3))


def status(history: str) -> str | None:
    cells = board(history).replace("\n", "")
    for mark, player in (("x", "X"), ("o", "O")):
        if any(all(cells[index] == mark for index in line) for line in WINS):
            return f"{player} wins"
    return "Draw" if len(history) == 9 else None


def main() -> None:
    from crystal9 import GameTokenizer
    from packed_int4 import PackedInt4Policy

    tokenizer = GameTokenizer.from_design_file(ROOT / "design.json")
    policy = PackedInt4Policy.load(ARTIFACT).eval()
    history = ""
    while not status(history):
        print(board(history))
        while True:
            move = input("Your move (a-i): ").strip().lower()
            if len(move) == 1 and move in SQUARES and move not in history:
                break
            print("Choose an empty square from a through i.")
        history += move
        if status(history):
            break
        move = policy.predict(history, tokenizer)
        if move not in SQUARES or move in history:
            raise RuntimeError(f"model returned {move!r} for {history!r}")
        history += move
        print(f"Crystal-9 chooses {move}.")
    print(board(history))
    print(status(history))


if __name__ == "__main__":
    main()
