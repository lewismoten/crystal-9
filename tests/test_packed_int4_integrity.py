import torch
import pytest

from crystal9 import TinyMoEPolicy
from packed_int4 import PackedInt4Policy, export_packed_int4


def test_packed_runtime_rejects_artifact_with_tampered_payload(tmp_path):
    torch.manual_seed(83)
    source = TinyMoEPolicy(vocab_size=13).eval()
    path = tmp_path / "model.pt"
    export_packed_int4(source, path, norm_weight_group_size=2)

    manifest = torch.load(path, map_location="cpu", weights_only=True)
    manifest["tensors"]["output.bias"]["packed"][0] ^= 1
    torch.save(manifest, path)

    with pytest.raises(ValueError, match="integrity"):
        PackedInt4Policy.load(path)
