import torch

from crystal9 import TinyMoEPolicy
from packed_int2 import PackedInt2Policy, export_packed_int2


def test_packed_int2_fp16_scales_match_fp16_scalar_materialization(tmp_path):
    torch.manual_seed(20260978)
    source = TinyMoEPolicy(vocab_size=13).eval()
    expected = TinyMoEPolicy(vocab_size=13).eval()
    expected.load_state_dict(source.state_dict())
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.copy_(parameter.sign() * parameter.abs().to(torch.float16).float())
    artifact_path = tmp_path / "crystal-9-int2-fp16-scales.pt"

    manifest = export_packed_int2(source, artifact_path, scale_dtype=torch.float16)
    runtime = PackedInt2Policy.load(artifact_path).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])

    torch.testing.assert_close(runtime(token_ids), expected(token_ids))
    assert manifest["scale_type"] == "float16"
