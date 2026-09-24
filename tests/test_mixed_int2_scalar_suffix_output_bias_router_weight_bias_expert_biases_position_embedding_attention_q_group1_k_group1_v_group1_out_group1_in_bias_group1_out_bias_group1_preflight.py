from pathlib import Path

import torch

import training
from crystal9 import GameTokenizer, TinyMoEPolicy


def test_int2_scalar_scope_through_attention_output_bias_preflight_records_zero_trainable_scope(tmp_path: Path):
    tokenizer = GameTokenizer.from_design_file("design.json")
    source = tmp_path / "f32-reference.pt"
    torch.save({"state_dict": TinyMoEPolicy(tokenizer.vocab_size).state_dict()}, source)

    report = training.run_mixed_int2_scalar_suffix_output_bias_router_weight_bias_expert_biases_position_embedding_attention_q_group1_k_group1_v_group1_out_group1_in_bias_group1_out_bias_group1_direct_preflight(
        source_path=source,
        output_dir=tmp_path / "preflight",
        histories=["", "a", "ab"],
        device=torch.device("cpu"),
    )

    assert report["layout"] == "mixed-int2-scalar-suffix-output-bias-router-weight-bias-expert-biases-position-embedding-attention-q-group1-k-group1-v-group1-out-group1-in-bias-group1-out-bias-group1-direct-materialization"
    assert report["trainable_tensors"] == []
    assert report["status"] in {"accepted-direct-materialization", "rejected-direct-materialization"}
    assert report["decision"] in {"advance", "change-strategy"}
    assert report["acceptance"]["legal_histories"] == 3
    assert (tmp_path / "preflight" / "report.json").is_file()
