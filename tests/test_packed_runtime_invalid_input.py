import torch

from crystal9 import GameTokenizer, TinyMoEPolicy
from packed_int4 import PackedInt4Policy, export_packed_int4


def test_packed_runtime_rejects_malformed_repeated_and_terminal_histories(tmp_path):
    torch.manual_seed(79)
    source = TinyMoEPolicy(vocab_size=13).eval()
    path = tmp_path / "model.pt"
    export_packed_int4(source, path, norm_weight_group_size=2)
    runtime = PackedInt4Policy.load(path)
    tokenizer = GameTokenizer.from_design_file("design.json")

    assert runtime.predict("z", tokenizer) == "!"
    assert runtime.predict("aa", tokenizer) == "!"
    assert runtime.predict("adbecf", tokenizer) == "!"  # X wins a-b-c
