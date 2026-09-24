from pathlib import Path

import training


SOURCE = Path("artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases-qat-200-seed20260945-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4-router-bias-expert-biases.pt")


def test_int3_group2_input_qat_records_only_input_tables_as_trainable(tmp_path: Path):
    report = training.run_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_input_group2_qat(
        epochs=0,
        histories=[""],
        source_path=SOURCE,
        output_dir=tmp_path,
        device=training.torch.device("cpu"),
    )

    assert report["trainable_tensors"] == ["embedding.weight", "position.weight"]
    assert report["frozen_tensor_sha256"]
