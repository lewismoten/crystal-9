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
    assert metadata["layout"] == "architecture-flow-v27"
    assert metadata["width"] == 1640
    assert metadata["legend_location"] == "top-right"
    assert metadata["execution_contract_panel"] == {
        "location": "bottom-left",
        "bounds": [20, 750, 570, 470],
        "header_lines": [
            "Proposed release: lewismoten/crystal-9:q4",
            "Decoded inspector — not reconstructable",
        ],
    }
    assert metadata["top_row_layout"] == {
        "attention_input_projection_x": 420,
        "attention_input_projection_group_bounds": [405, 30, 200, 650],
        "attention_output_projection_x": 675,
        "attention_output_projection_label": "Attention output projection",
        "attention_output_projection_display_lines": ["Attention output", "projection"],
        "norm_x": 895,
        "router_x": 995,
        "final_output_x": 1205,
        "final_output_return_x": 1285,
        "post_position_arrow": {"start_x": 375, "end_x": 404},
        "attention_input_to_output_arrow": {"start_x": 625, "end_x": 655},
    }
    assert metadata["attention_input_projection"] == {
        "group_label": "Attention input projections",
        "packed_weight_shape": [96, 32],
        "segments": {
            "Query": {"weight_shape": [32, 32], "bias_shape": [32]},
            "Key": {"weight_shape": [32, 32], "bias_shape": [32]},
            "Value": {"weight_shape": [32, 32], "bias_shape": [32]},
        },
    }
    assert metadata["embedding_combination"] == {
        "operation": "elementwise addition",
        "inputs": ["token embedding", "position embedding"],
        "output": "attention input",
    }
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
