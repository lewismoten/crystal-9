import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_design():
    return json.loads((ROOT / "design.json").read_text())


def test_32_wide_architecture_is_block_quantization_oriented():
    design = load_design()
    architecture = design["architecture"]

    assert architecture == {
        "hidden_size": 32,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_experts": 9,
        "num_experts_per_token": 2,
        "context_window": 8,
    }


def test_custom_vocabulary_is_explicit_and_only_accepts_game_symbols():
    design = load_design()
    vocabulary = design["vocabulary"]

    assert vocabulary["model"] == "crystal-9-game-symbols-v1"
    assert vocabulary["tokens"] == ["<pad>", "<bos>", "<eos>", "!", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
    assert vocabulary["input_symbols"] == list("abcdefghi")
    assert vocabulary["invalid_output"] == "!"
