import torch

from crystal9 import TinyMoEPolicy
from packed_int4 import PackedInt4Policy, export_packed_int4


def test_packed_int4_can_store_scales_as_fp16_and_run(tmp_path):
    torch.manual_seed(97)
    source = TinyMoEPolicy(vocab_size=13).eval()
    fp32_path = tmp_path / "fp32-scales.pt"
    fp16_path = tmp_path / "fp16-scales.pt"

    export_packed_int4(source, fp32_path, norm_weight_group_size=2)
    manifest = export_packed_int4(source, fp16_path, norm_weight_group_size=2, scale_storage="float16")
    loaded = torch.load(fp16_path, map_location="cpu", weights_only=True)
    runtime = PackedInt4Policy.load(fp16_path).eval()

    assert manifest["format"] == "crystal-9-packed-int4-fp16-scales-v1"
    assert manifest["scale_storage"] == "float16"
    assert {record["scales"].dtype for record in loaded["tensors"].values()} == {torch.float16}
    assert fp16_path.stat().st_size < fp32_path.stat().st_size
    assert torch.isfinite(runtime(torch.tensor([[1, 2, 3, 0]])).float()).all()
