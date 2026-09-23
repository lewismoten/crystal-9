import torch

from crystal9 import TinyMoEPolicy, materialize_mixed_int4_full_parameters_grouped_norm_weight
from packed_int4 import PackedInt4Policy, export_packed_int4


def test_packed_int4_runtime_matches_materialized_full_layout(tmp_path):
    torch.manual_seed(67)
    source = TinyMoEPolicy(vocab_size=13).eval()
    artifact_path = tmp_path / "crystal-9-int4.pt"
    manifest = export_packed_int4(source, artifact_path, norm_weight_group_size=2)

    runtime = PackedInt4Policy.load(artifact_path).eval()
    token_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    expected = materialize_mixed_int4_full_parameters_grouped_norm_weight(source, 2)(token_ids)

    torch.testing.assert_close(runtime(token_ids), expected)
    assert manifest["format"] == "crystal-9-packed-int4-v1"
    assert manifest["parameter_values"] == sum(parameter.numel() for parameter in source.parameters())
    assert artifact_path.exists()
