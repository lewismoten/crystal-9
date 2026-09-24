import hashlib
from pathlib import Path

from tools.tensor_inspector import render_checkpoint_inspector


def test_checkpoint_inspector_renders_actual_tensor_inventory_as_png():
    source = Path("artifacts-fp32.pt")

    png, metadata = render_checkpoint_inspector(source)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["tensor_count"] == 48
    assert metadata["tensors"]["attention.in_proj_weight"]["shape"] == [96, 32]
    assert metadata["tensors"]["experts.0.0.weight"]["shape"] == [32, 32]
    assert metadata["normalization"] == "per-tensor symmetric max-absolute"
    assert metadata["layout"] == "architecture-flow-v19"
    assert metadata["bias_alignment"] == "vertical output-row axis"
    assert metadata["sections"][-2:] == ["experts", "output"]
    assert metadata["legend"]["B"] == "bias column; one value per output row"
    assert metadata["expert_layout"] == "five Expert boxes over four Expert boxes; each contains its first and second layer"
    assert metadata["calculation_flow"] == ["embeddings and positions", "attention", "norm and router", "top-2 routed experts", "combined output"]
    assert metadata["vocabulary"] == ["<pad>", "<bos>", "<eos>", "!", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
    assert metadata["font"] == "DejaVu Sans"
    assert metadata["router_selection_label"] == "Selected: 2 experts"
    assert metadata["router_selection_lines"] == ["Selected:", "2 experts"]
    assert metadata["router_selection_shape"] == "gray outlined rectangle"
    assert metadata["router_path"] == "continuous downward arrow touching the Experts outline"
    assert metadata["expert_return_path"] == "straight upward arrow ends below Final output matrix"
    assert metadata["intra_expert_arrows"] == "compact clear gap between first and second layer matrices"
    assert metadata["border_legend"] == {"dim purple": "weight matrix", "dim cyan": "bias vector"}
    assert metadata["value_legend"] == {"yellow": "large positive", "green": "moderate positive", "black": "neutral / zero", "blue": "negative", "bright blue": "large negative"}
    assert metadata["experts_outline"] == "slate gray"
    assert metadata["representation"] == "decoded inspector; not reconstructable"
    assert metadata["proposed_deployment_tag"] == "lewismoten/crystal-9:q4"
    assert metadata["execution_contract"] == {
        "public_input": "a-i; maximum 8 moves",
        "sequence": "BOS + history; PAD to 9 positions",
        "attention": "causal mask; read final non-PAD state",
        "routing": "softmax router; top 2 of 9 experts",
        "expert": "32 -> 32 SiLU -> 32",
        "public_output": "a-i; ! is invalid/no-move sentinel",
    }
