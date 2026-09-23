import torch

from crystal9 import GameTokenizer
from training import evaluate


class Probe:
    def __init__(self):
        self.received = None

    def eval(self):
        return self

    def forward_mixed_int4_input_attention_groups(self, token_ids, groups, quantize_attention_biases=False):
        self.received = (groups, quantize_attention_biases)
        logits = torch.zeros((token_ids.shape[0], 13), device=token_ids.device)
        logits[:, 5] = 1
        return logits


def test_evaluate_passes_attention_bias_quantization_to_fake_runtime():
    tokenizer = GameTokenizer.from_design_file("design.json")
    model = Probe()
    groups = frozenset({"q", "k", "v", "out"})

    evaluate(model, tokenizer, torch.device("cpu"), histories=[""], attention_int4_groups=groups, quantize_attention_biases=True)

    assert model.received == (groups, True)
