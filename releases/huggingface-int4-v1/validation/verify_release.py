"""Validate every staged Crystal-9 release artifact against the policy oracle."""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import torch

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from crystal9 import GameTokenizer, TinyMoEPolicy  # noqa: E402
from packed_int4 import PackedInt4Policy, evaluate_packed  # noqa: E402

SQUARES = "abcdefghi"
MOVE_ORDER = "ebdfhcgia"
WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
EXPECTED = {"legal_histories": 294778, "policy_misses": 0}
INVALID = ("!", "aa", "abcdefghi", "adbecf")


def winner(board: str) -> str | None:
    for a, b, c in WINS:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None


def board_for(history: str) -> str | None:
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
    won = winner(board)
    if won == "X":
        return 1
    if won == "O":
        return -1
    if "." not in board:
        return 0
    values = [score(board[:square] + turn + board[square + 1 :], "O" if turn == "X" else "X") for square in range(9) if board[square] == "."]
    return max(values) if turn == "X" else min(values)


def optimal_move(history: str) -> str:
    board = board_for(history)
    if board is None or winner(board) or "." not in board:
        return "!"
    turn = "X" if len(history) % 2 == 0 else "O"
    candidates = [(score(board[:SQUARES.index(symbol)] + turn + board[SQUARES.index(symbol) + 1 :], "O" if turn == "X" else "X"), symbol) for symbol in MOVE_ORDER if board[SQUARES.index(symbol)] == "."]
    best = max(value for value, _ in candidates) if turn == "X" else min(value for value, _ in candidates)
    return next(symbol for value, symbol in candidates if value == best)


def legal_histories(prefix: str = "") -> list[str]:
    board = board_for(prefix)
    if board is None or winner(board) or len(prefix) == 8:
        return [prefix]
    return [prefix] + [child for symbol in SQUARES if board[SQUARES.index(symbol)] == "." for child in legal_histories(prefix + symbol)]


def padded(tokenizer: GameTokenizer, history: str) -> list[int]:
    tokens = tokenizer.encode_history(history)
    return tokens + [0] * (9 - len(tokens))


def evaluate_f32(model: TinyMoEPolicy, tokenizer: GameTokenizer, histories: list[str]) -> dict[str, int]:
    misses = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(histories), 4096):
            batch = histories[start : start + 4096]
            logits = model(torch.tensor([padded(tokenizer, history) for history in batch]))
            misses += sum(tokenizer.decode_id(token) != optimal_move(history) for token, history in zip(logits.argmax(dim=-1).tolist(), batch))
    return {"legal_histories": len(histories), "policy_misses": misses}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(result: dict[str, int], label: str) -> None:
    if result != EXPECTED:
        raise SystemExit(f"{label} exhaustive policy gate failed: {result}")


def main() -> None:
    tokenizer = GameTokenizer.from_design_file(STAGE / "design.json")
    histories = [history for history in legal_histories() if optimal_move(history) != "!"]

    f32_path = STAGE / "artifacts/artifacts-fp32.pt"
    checkpoint = torch.load(f32_path, map_location="cpu", weights_only=False)
    f32 = TinyMoEPolicy(tokenizer.vocab_size)
    f32.load_state_dict(checkpoint["state_dict"])
    f32_result = evaluate_f32(f32, tokenizer, histories)
    require(f32_result, "F32 reference")

    records = [{
        "id": "f32-reference-v1",
        "path": "artifacts/artifacts-fp32.pt",
        "format": "PyTorch F32 state_dict",
        "bytes": f32_path.stat().st_size,
        "sha256": digest(f32_path),
        "runtime": "TinyMoEPolicy",
        "acceptance": f32_result,
    }]
    for identifier, filename in (
        ("packed-int4-fp32-scales-v1", "crystal-9-int4-group2-packed-v1.pt"),
        ("packed-int4-fp16-scales-v1", "crystal-9-int4-group2-packed-fp16-scales-v1.pt"),
    ):
        path = STAGE / "artifacts" / filename
        runtime = PackedInt4Policy.load(path).eval()
        acceptance = evaluate_packed(runtime, tokenizer, torch.device("cpu"), histories, optimal_move)
        require(acceptance, identifier)
        invalid = {history: runtime.predict(history, tokenizer) for history in INVALID}
        if any(result != "!" for result in invalid.values()):
            raise SystemExit(f"{identifier} invalid-history gate failed: {invalid}")
        records.append({
            "id": identifier,
            "path": f"artifacts/{filename}",
            "format": runtime.manifest["format"],
            "scale_storage": runtime.manifest.get("scale_storage", "float32"),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "integrity_sha256": runtime.manifest["integrity_sha256"],
            "runtime": "PackedInt4Policy",
            "acceptance": acceptance,
            "invalid_input_examples": invalid,
        })

    report = {"release_id": "crystal-9-accepted-artifacts-v1", "artifacts": records}
    (STAGE / "validation/release-acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
