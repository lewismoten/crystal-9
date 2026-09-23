"""Train and precision-gate Crystal-9's custom-vocabulary policy model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from crystal9 import GameTokenizer, TinyMoEPolicy, quantize_tensor

SQUARES = "abcdefghi"
MOVE_ORDER = "ebdfhcgia"
WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


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
    values = []
    for square in range(9):
        if board[square] == ".":
            next_board = board[:square] + turn + board[square + 1 :]
            values.append(score(next_board, "O" if turn == "X" else "X"))
    return max(values) if turn == "X" else min(values)


def optimal_move(history: str) -> str:
    board = board_for(history)
    if board is None or winner(board) or "." not in board:
        return "!"
    turn = "X" if len(history) % 2 == 0 else "O"
    candidates = []
    for symbol in MOVE_ORDER:
        square = SQUARES.index(symbol)
        if board[square] == ".":
            candidate = board[:square] + turn + board[square + 1 :]
            candidates.append((score(candidate, "O" if turn == "X" else "X"), symbol))
    best = max(value for value, _ in candidates) if turn == "X" else min(value for value, _ in candidates)
    return next(symbol for value, symbol in candidates if value == best)


def legal_histories(prefix: str = "") -> list[str]:
    board = board_for(prefix)
    if board is None or winner(board) or len(prefix) == 8:
        return [prefix]
    histories = [prefix]
    for symbol in SQUARES:
        if board[SQUARES.index(symbol)] == ".":
            histories.extend(legal_histories(prefix + symbol))
    return histories


def padded(tokenizer: GameTokenizer, history: str) -> list[int]:
    tokens = tokenizer.encode_history(history)
    return tokens + [0] * (9 - len(tokens))


def evaluate(model: TinyMoEPolicy, tokenizer: GameTokenizer, device: torch.device) -> dict[str, int]:
    model.eval()
    histories = [history for history in legal_histories() if optimal_move(history) != "!"]
    misses = 0
    with torch.no_grad():
        for start in range(0, len(histories), 4096):
            batch = histories[start : start + 4096]
            logits = model(torch.tensor([padded(tokenizer, history) for history in batch], device=device))
            predicted = logits.argmax(dim=-1).tolist()
            misses += sum(tokenizer.decode_id(token_id) != optimal_move(history) for token_id, history in zip(predicted, batch))
    return {"legal_histories": len(histories), "policy_misses": misses}


def run(epochs: int = 400) -> dict:
    root = Path(__file__).parent
    tokenizer = GameTokenizer.from_design_file(root / "design.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    histories = [history for history in legal_histories() if optimal_move(history) != "!"]
    inputs = torch.tensor([padded(tokenizer, history) for history in histories], device=device)
    labels = torch.tensor([tokenizer.tokens.index(optimal_move(history)) for history in histories], device=device)
    torch.manual_seed(9)
    model = TinyMoEPolicy(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), labels)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            report = {"epoch": epoch, "loss": float(loss.item()), "device": str(device), "examples": len(histories)}
            (root / "training-progress.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"state_dict": model.cpu().state_dict(), "design": json.loads((root / "design.json").read_text())}, root / "artifacts-fp32.pt")
    results = {"fp32": evaluate(model.to(device), tokenizer, device)}
    for bits in (16, 8, 6, 4, 3, 2, 1):
        candidate = TinyMoEPolicy(tokenizer.vocab_size).to(device)
        candidate.load_state_dict(model.to(device).state_dict())
        with torch.no_grad():
            for parameter in candidate.parameters():
                parameter.copy_(quantize_tensor(parameter, bits))
        results[f"int{bits}"] = evaluate(candidate, tokenizer, device)
    (root / "precision-report.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
