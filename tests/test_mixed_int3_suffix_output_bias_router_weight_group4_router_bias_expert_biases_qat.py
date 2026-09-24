from pathlib import Path

import training


def test_int3_group4_router_and_expert_biases_qat_records_only_expert_biases_as_trainable(tmp_path: Path):
    report = training.run_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_expert_biases_qat(
        epochs=0,
        histories=[""],
        source_path=Path("artifacts/int3-suffix-output-bias-router-weight-group4-router-bias-qat-200-seed20260944-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4-router-bias.pt"),
        output_dir=tmp_path,
        device=training.torch.device("cpu"),
    )

    assert report["trainable_tensors"] == ["experts.*.0.bias", "experts.*.2.bias"]
    assert report["frozen_tensor_sha256"]
