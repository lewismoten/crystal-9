from pathlib import Path

import training


def test_int3_group4_router_weight_router_bias_qat_records_isolated_trainable_scope(tmp_path: Path):
    report = training.run_mixed_int3_suffix_output_bias_router_weight_group4_router_bias_qat(
        epochs=0,
        histories=[""],
        source_path=Path("artifacts/int3-suffix-output-bias-router-weight-group4-qat-200-seed20260943-lr1e-4/artifacts-qat-mixed-int3-suffix-output-bias-router-weight-group4.pt"),
        output_dir=tmp_path,
        device=training.torch.device("cpu"),
    )

    assert report["trainable_tensors"] == ["router.bias"]
    assert report["frozen_tensor_sha256"]
