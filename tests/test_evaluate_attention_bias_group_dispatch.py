import torch

from crystal9 import GameTokenizer
from training import evaluate


class Probe:
    def __init__(self):
        self.received = None

    def eval(self):
        return self

    def forward_mixed_int4_input_attention_groups(self, token_ids, groups, quantize_attention_biases=False, attention_bias_groups=None):
        self.received = (groups, quantize_attention_biases, attention_bias_groups)
        logits = torch.zeros((token_ids.shape[0], 13), device=token_ids.device)
        logits[:, 5] = 1
        return logits


def test_evaluate_passes_selected_attention_bias_groups_to_fake_runtime():
    tokenizer = GameTokenizer.from_design_file("design.json")
    model = Probe()
    groups = frozenset({"q", "k", "v", "out"})
    bias_groups = frozenset({"out"})

    evaluate(
        model,
        tokenizer,
        torch.device("cpu"),
        histories=[""],
        attention_int4_groups=groups,
        quantize_attention_biases=True,
        attention_bias_groups=bias_groups,
    )

    assert model.received == (groups, True, bias_groups)
