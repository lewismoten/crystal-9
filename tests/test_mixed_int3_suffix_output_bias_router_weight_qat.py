from pathlib import Path

import training


def test_int3_router_weight_qat_records_isolated_trainable_scope(tmp_path: Path):
    report = training.run_mixed_int3_suffix_output_bias_router_weight_qat(
        epochs=0,
        histories=[""],
        source_path=Path("artifacts/int3-suffix-qat-300-continuation-seed20260936-lr5e-5/artifacts-qat-mixed-int3-suffix.pt"),
        output_dir=tmp_path,
        device=training.torch.device("cpu"),
    )

    assert report["trainable_tensors"] == ["router.weight"]
    assert report["frozen_tensor_sha256"]
