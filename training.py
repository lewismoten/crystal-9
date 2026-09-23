"""Train and precision-gate Crystal-9's custom-vocabulary policy model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from crystal9 import GameTokenizer, TinyMoEPolicy, materialize_mixed_int4, materialize_mixed_int4_input, materialize_mixed_int4_input_attention, materialize_mixed_int4_input_attention_q, materialize_mixed_int4_input_attention_v, quantize_tensor

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


def evaluate(
    model: TinyMoEPolicy,
    tokenizer: GameTokenizer,
    device: torch.device,
    histories: list[str] | None = None,
    qat_bits: int | None = None,
    mixed_int4: bool = False,
    mixed_int4_input: bool = False,
    mixed_int4_input_attention: bool = False,
    attention_int4_groups: frozenset[str] | None = None,
) -> dict[str, int]:
    model.eval()
    histories = histories if histories is not None else [history for history in legal_histories() if optimal_move(history) != "!"]
    misses = 0
    with torch.no_grad():
        for start in range(0, len(histories), 4096):
            batch = histories[start : start + 4096]
            inputs = torch.tensor([padded(tokenizer, history) for history in batch], device=device)
            logits = model.forward_mixed_int4_input_attention_groups(inputs, attention_int4_groups) if attention_int4_groups is not None else model.forward_mixed_int4_input_attention(inputs) if mixed_int4_input_attention else model.forward_mixed_int4_input(inputs) if mixed_int4_input else model.forward_mixed_int4(inputs) if mixed_int4 else model.forward_quantized(inputs, qat_bits) if qat_bits else model(inputs)
            predicted = logits.argmax(dim=-1).tolist()
            misses += sum(tokenizer.decode_id(token_id) != optimal_move(history) for token_id, history in zip(predicted, batch))
    return {"legal_histories": len(histories), "policy_misses": misses}


def load_reference_model(path: str | Path, vocab_size: int, device: torch.device) -> TinyMoEPolicy:
    """Load a compatible F32 checkpoint as master weights for QAT."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TinyMoEPolicy(vocab_size).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def qat_step(
    model: TinyMoEPolicy,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    bits: int,
) -> float:
    """One master-weight update through the fake-quantized forward path."""
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model.forward_quantized(inputs, bits), labels)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def batch_ranges(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        yield start, min(total, start + batch_size)


def run_mixed_int4_input_attention_v_qat(
    epochs: int = 200,
    batch_size: int = 1024,
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    histories: list[str] | None = None,
    device: torch.device | None = None,
) -> dict:
    """Add the V projection to the accepted Q INT4 attention stage."""
    root = Path(__file__).parent
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = GameTokenizer.from_design_file(root / "design.json")
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = Path(source_path) if source_path else root / "artifacts/mixed-int4-row-input-attention-q-baseline/artifacts-qat-mixed-int4-row-input-attention-q.pt"
    histories = histories or [history for history in legal_histories() if optimal_move(history) != "!"]
    inputs = torch.tensor([padded(tokenizer, history) for history in histories], device=device)
    labels = torch.tensor([tokenizer.tokens.index(optimal_move(history)) for history in histories], device=device)
    model = load_reference_model(source, tokenizer.vocab_size, device)
    for parameter in list(model.experts.parameters()) + list(model.output.parameters()):
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0008, weight_decay=0.0001)
    groups = frozenset({"q", "v"})
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(histories), device=device)
        weighted_loss = 0.0
        for start, end in batch_ranges(len(histories), batch_size):
            index = order[start:end]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model.forward_mixed_int4_input_attention_groups(inputs[index], groups), labels[index])
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(index)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            progress = {"epoch": epoch, "layout": "mixed-int4-row-input-attention-q-v", "loss": weighted_loss / len(histories), "device": str(device), "examples": len(histories), "batch_size": batch_size, "source_checkpoint": source.name, "frozen_suffix": ["experts", "output"]}
            (output / "mixed-int4-row-input-attention-q-v-progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    checkpoint_path = output / "artifacts-qat-mixed-int4-row-input-attention-q-v.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "layout": "mixed-int4-row-input-attention-q-v", "source_checkpoint": source.name, "frozen_suffix": ["experts", "output"]}, checkpoint_path)
    model = model.to(device)
    materialized = materialize_mixed_int4_input_attention_v(model)
    report = {
        "layout": "mixed-int4-row-input-attention-q-v",
        "frozen_suffix": ["experts", "output"],
        "qat_forward": evaluate(model, tokenizer, device, histories, attention_int4_groups=groups),
        "materialized_mixed_int4_row_input_attention_q_v": evaluate(materialized, tokenizer, device, histories),
        "source_checkpoint": source.name,
    }
    (output / "mixed-int4-row-input-attention-q-v-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_mixed_int4_input_attention_q_qat(
    epochs: int = 200,
    batch_size: int = 1024,
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    histories: list[str] | None = None,
    device: torch.device | None = None,
) -> dict:
    """Add only the Q attention projection to the verified input-INT4 layout."""
    root = Path(__file__).parent
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = GameTokenizer.from_design_file(root / "design.json")
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = Path(source_path) if source_path else root / "artifacts/mixed-int4-row-input-continue-800/artifacts-qat-mixed-int4-row-input.pt"
    histories = histories or [history for history in legal_histories() if optimal_move(history) != "!"]
    inputs = torch.tensor([padded(tokenizer, history) for history in histories], device=device)
    labels = torch.tensor([tokenizer.tokens.index(optimal_move(history)) for history in histories], device=device)
    model = load_reference_model(source, tokenizer.vocab_size, device)
    for parameter in list(model.experts.parameters()) + list(model.output.parameters()):
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0008, weight_decay=0.0001)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(histories), device=device)
        weighted_loss = 0.0
        for start, end in batch_ranges(len(histories), batch_size):
            index = order[start:end]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model.forward_mixed_int4_input_attention_q(inputs[index]), labels[index])
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(index)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            progress = {"epoch": epoch, "layout": "mixed-int4-row-input-attention-q", "loss": weighted_loss / len(histories), "device": str(device), "examples": len(histories), "batch_size": batch_size, "source_checkpoint": source.name, "frozen_suffix": ["experts", "output"]}
            (output / "mixed-int4-row-input-attention-q-progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    checkpoint_path = output / "artifacts-qat-mixed-int4-row-input-attention-q.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "layout": "mixed-int4-row-input-attention-q", "source_checkpoint": source.name, "frozen_suffix": ["experts", "output"]}, checkpoint_path)
    model = model.to(device)
    materialized = materialize_mixed_int4_input_attention_q(model)
    report = {
        "layout": "mixed-int4-row-input-attention-q",
        "frozen_suffix": ["experts", "output"],
        "qat_forward": evaluate(model, tokenizer, device, histories, attention_int4_groups=frozenset({"q"})),
        "materialized_mixed_int4_row_input_attention_q": evaluate(materialized, tokenizer, device, histories),
        "source_checkpoint": source.name,
    }
    (output / "mixed-int4-row-input-attention-q-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_mixed_int4_input_attention_qat(
    epochs: int = 200,
    batch_size: int = 1024,
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    histories: list[str] | None = None,
    device: torch.device | None = None,
) -> dict:
    """Add INT4 attention projections while preserving the expert/output suffix."""
    root = Path(__file__).parent
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = GameTokenizer.from_design_file(root / "design.json")
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = Path(source_path) if source_path else root / "artifacts-qat-mixed-int4-row-input.pt"
    histories = histories or [history for history in legal_histories() if optimal_move(history) != "!"]
    inputs = torch.tensor([padded(tokenizer, history) for history in histories], device=device)
    labels = torch.tensor([tokenizer.tokens.index(optimal_move(history)) for history in histories], device=device)
    model = load_reference_model(source, tokenizer.vocab_size, device)
    for parameter in list(model.experts.parameters()) + list(model.output.parameters()):
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0008, weight_decay=0.0001)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(histories), device=device)
        weighted_loss = 0.0
        for start, end in batch_ranges(len(histories), batch_size):
            index = order[start:end]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model.forward_mixed_int4_input_attention(inputs[index]), labels[index])
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(index)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            progress = {"epoch": epoch, "layout": "mixed-int4-row-input-attention", "loss": weighted_loss / len(histories), "device": str(device), "examples": len(histories), "batch_size": batch_size, "source_checkpoint": source.name, "frozen_suffix": ["experts", "output"]}
            (output / "mixed-int4-row-input-attention-progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    checkpoint_path = output / "artifacts-qat-mixed-int4-row-input-attention.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "layout": "mixed-int4-row-input-attention", "source_checkpoint": source.name, "frozen_suffix": ["experts", "output"]}, checkpoint_path)
    model = model.to(device)
    materialized = materialize_mixed_int4_input_attention(model)
    report = {
        "layout": "mixed-int4-row-input-attention",
        "frozen_suffix": ["experts", "output"],
        "qat_forward": evaluate(model, tokenizer, device, histories, mixed_int4_input_attention=True),
        "materialized_mixed_int4_row_input_attention": evaluate(materialized, tokenizer, device, histories),
        "source_checkpoint": source.name,
    }
    (output / "mixed-int4-row-input-attention-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_mixed_int4_input_qat(
    epochs: int = 200,
    batch_size: int = 1024,
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    histories: list[str] | None = None,
    device: torch.device | None = None,
) -> dict:
    """Add INT4 input tables while freezing the verified expert/output suffix."""
    root = Path(__file__).parent
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = GameTokenizer.from_design_file(root / "design.json")
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = Path(source_path) if source_path else root / "artifacts-qat-mixed-int4-row.pt"
    histories = histories or [history for history in legal_histories() if optimal_move(history) != "!"]
    inputs = torch.tensor([padded(tokenizer, history) for history in histories], device=device)
    labels = torch.tensor([tokenizer.tokens.index(optimal_move(history)) for history in histories], device=device)
    model = load_reference_model(source, tokenizer.vocab_size, device)
    for parameter in list(model.experts.parameters()) + list(model.output.parameters()):
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0008, weight_decay=0.0001)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(histories), device=device)
        weighted_loss = 0.0
        for start, end in batch_ranges(len(histories), batch_size):
            index = order[start:end]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model.forward_mixed_int4_input(inputs[index]), labels[index])
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(index)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            progress = {"epoch": epoch, "layout": "mixed-int4-row-input", "loss": weighted_loss / len(histories), "device": str(device), "examples": len(histories), "batch_size": batch_size, "source_checkpoint": source.name, "frozen_suffix": ["experts", "output"]}
            (output / "mixed-int4-row-input-progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    checkpoint_path = output / "artifacts-qat-mixed-int4-row-input.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "layout": "mixed-int4-row-input", "source_checkpoint": source.name, "frozen_suffix": ["experts", "output"]}, checkpoint_path)
    model = model.to(device)
    materialized = materialize_mixed_int4_input(model)
    report = {
        "layout": "mixed-int4-row-input",
        "frozen_suffix": ["experts", "output"],
        "qat_forward": evaluate(model, tokenizer, device, histories, mixed_int4_input=True),
        "materialized_mixed_int4_row_input": evaluate(materialized, tokenizer, device, histories),
        "source_checkpoint": source.name,
    }
    (output / "mixed-int4-row-input-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_mixed_int4_qat(
    epochs: int = 200,
    batch_size: int = 1024,
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    histories: list[str] | None = None,
    device: torch.device | None = None,
) -> dict:
    """Train and gate one named, identical mixed-INT4-row layout."""
    root = Path(__file__).parent
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = GameTokenizer.from_design_file(root / "design.json")
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = Path(source_path) if source_path else root / "artifacts-fp32.pt"
    histories = histories or [history for history in legal_histories() if optimal_move(history) != "!"]
    inputs = torch.tensor([padded(tokenizer, history) for history in histories], device=device)
    labels = torch.tensor([tokenizer.tokens.index(optimal_move(history)) for history in histories], device=device)
    model = load_reference_model(source, tokenizer.vocab_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0001)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(histories), device=device)
        weighted_loss = 0.0
        for start, end in batch_ranges(len(histories), batch_size):
            index = order[start:end]
            optimizer.zero_grad()
            mixed_loss = nn.functional.cross_entropy(model.forward_mixed_int4(inputs[index]), labels[index])
            reference_loss = nn.functional.cross_entropy(model(inputs[index]), labels[index])
            loss = 0.75 * mixed_loss + 0.25 * reference_loss
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(index)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            progress = {
                "epoch": epoch,
                "layout": "mixed-int4-row",
                "loss": weighted_loss / len(histories),
                "device": str(device),
                "examples": len(histories),
                "batch_size": batch_size,
                "source_checkpoint": source.name,
            }
            (output / "mixed-int4-row-progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    checkpoint_path = output / "artifacts-qat-mixed-int4-row.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "layout": "mixed-int4-row", "source_checkpoint": source.name}, checkpoint_path)
    model = model.to(device)
    materialized = materialize_mixed_int4(model)
    report = {
        "layout": "mixed-int4-row",
        "fp32_master": evaluate(model, tokenizer, device, histories),
        "qat_forward": evaluate(model, tokenizer, device, histories, mixed_int4=True),
        "materialized_mixed_int4": evaluate(materialized, tokenizer, device, histories),
        "source_checkpoint": source.name,
    }
    (output / "mixed-int4-row-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_qat(
    bits: int,
    epochs: int = 200,
    batch_size: int = 1024,
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    histories: list[str] | None = None,
    device: torch.device | None = None,
) -> dict:
    """Fine-tune master weights through a selected low-bit forward path."""
    root = Path(__file__).parent
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = GameTokenizer.from_design_file(root / "design.json")
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = Path(source_path) if source_path else root / "artifacts-fp32.pt"
    histories = histories or [history for history in legal_histories() if optimal_move(history) != "!"]
    inputs = torch.tensor([padded(tokenizer, history) for history in histories], device=device)
    labels = torch.tensor([tokenizer.tokens.index(optimal_move(history)) for history in histories], device=device)
    model = load_reference_model(source, tokenizer.vocab_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0001)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(histories), device=device)
        weighted_loss = 0.0
        for start, end in batch_ranges(len(histories), batch_size):
            index = order[start:end]
            optimizer.zero_grad()
            qat_loss = nn.functional.cross_entropy(model.forward_quantized(inputs[index], bits), labels[index])
            reference_loss = nn.functional.cross_entropy(model(inputs[index]), labels[index])
            loss = 0.75 * qat_loss + 0.25 * reference_loss
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(index)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            progress = {
                "epoch": epoch,
                "bits": bits,
                "loss": weighted_loss / len(histories),
                "device": str(device),
                "examples": len(histories),
                "batch_size": batch_size,
                "source_checkpoint": source.name,
            }
            (output / f"qat-int{bits}-progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    checkpoint_path = output / f"artifacts-qat-int{bits}.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "bits": bits, "source_checkpoint": source.name}, checkpoint_path)
    model = model.to(device)
    post_training = TinyMoEPolicy(tokenizer.vocab_size).to(device)
    post_training.load_state_dict(model.state_dict())
    with torch.no_grad():
        for parameter in post_training.parameters():
            parameter.copy_(quantize_tensor(parameter, bits))
    report = {
        "bits": bits,
        "fp32_master": evaluate(model, tokenizer, device, histories),
        "qat_forward": evaluate(model, tokenizer, device, histories, qat_bits=bits),
        "post_training_quantized": evaluate(post_training, tokenizer, device, histories),
        "source_checkpoint": source.name,
    }
    (output / f"qat-int{bits}-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(epochs: int = 400, batch_size: int = 1024) -> dict:
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
        order = torch.randperm(len(histories), device=device)
        weighted_loss = 0.0
        for start, end in batch_ranges(len(histories), batch_size):
            index = order[start:end]
            optimizer.zero_grad()
            loss = loss_fn(model(inputs[index]), labels[index])
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(index)
        loss_value = weighted_loss / len(histories)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            report = {"epoch": epoch, "loss": loss_value, "device": str(device), "examples": len(histories), "batch_size": batch_size}
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
