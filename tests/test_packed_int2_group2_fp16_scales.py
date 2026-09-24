import torch

from crystal9 import TinyMoEPolicy
from packed_int2_group2_fp16_scales import PackedInt2Group2FP16ScalePolicy, export_packed_int2_group2_fp16_scales


def test_packed_int2_group2_fp16_scale_runtime_matches_groupwise_materialization(tmp_path):
    torch.manual_seed(20260981)
    source = TinyMoEPolicy(vocab_size=13).eval()
    artifact_path = tmp_path / "crystal-9-int2-group2-fp16-scales.pt"

    manifest = export_packed_int2_group2_fp16_scales(source, artifact_path)
    runtime = PackedInt2Group2FP16ScalePolicy.load(artifact_path).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])

    expected = TinyMoEPolicy(vocab_size=13).eval()
    expected.load_state_dict(source.state_dict())
    with torch.no_grad():
        for parameter in expected.parameters():
            values = parameter.reshape(-1)
            padded = torch.nn.functional.pad(values, (0, (-values.numel()) % 2))
            raw_scales = padded.reshape(-1, 2).abs().amax(dim=1).repeat_interleave(2)[: values.numel()]
            stored_scales = raw_scales.to(torch.float16).float()
            codes = (values / torch.where(raw_scales == 0, torch.ones_like(raw_scales), raw_scales)).round().clamp(-1, 1)
            values.copy_(codes * stored_scales)

    torch.testing.assert_close(runtime(token_ids), expected(token_ids))
    assert manifest["scale_count"] == sum((value.numel() + 1) // 2 for value in source.state_dict().values())
    assert manifest["group_size"] == 2
    assert manifest["scale_type"] == "float16"
