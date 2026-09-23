import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from packed_int4 import PackedInt4Policy, evaluate_packed, export_packed_int4
from training import optimal_move, padded


def test_packed_runtime_evaluator_reports_policy_misses(tmp_path):
    torch.manual_seed(71)
    source = TinyMoEPolicy(vocab_size=13).eval()
    path = tmp_path / "model.pt"
    export_packed_int4(source, path, norm_weight_group_size=2)
    tokenizer = GameTokenizer.from_design_file("design.json")
    histories = ["", "e", "ea"]

    runtime = PackedInt4Policy.load(path)
    report = evaluate_packed(runtime, tokenizer, torch.device("cpu"), histories, optimal_move)
    expected_misses = 0
    for history in histories:
        token_ids = torch.tensor([padded(tokenizer, history)])
        expected_misses += tokenizer.decode_id(runtime(token_ids).argmax().item()) != optimal_move(history)

    assert report == {"legal_histories": 3, "policy_misses": expected_misses}
