import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from packed_int3 import PackedInt3Policy, evaluate_packed, export_packed_int3
from training import optimal_move


def test_packed_int3_evaluator_handles_longest_legal_history(tmp_path):
    source = TinyMoEPolicy(vocab_size=13).eval()
    path = tmp_path / "crystal-9-int3.pt"
    export_packed_int3(source, path)
    tokenizer = GameTokenizer.from_design_file("design.json")

    report = evaluate_packed(PackedInt3Policy.load(path), tokenizer, torch.device("cpu"), ["aebdcfgh"], optimal_move)

    assert report["legal_histories"] == 1
