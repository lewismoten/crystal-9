import pytest
import torch

from crystal9 import TinyMoEPolicy
from packed_int3 import PackedInt3Policy, export_packed_int3


def test_packed_int3_runtime_rejects_tampered_payload(tmp_path):
    torch.manual_seed(103)
    source = TinyMoEPolicy(vocab_size=13).eval()
    path = tmp_path / "crystal-9-int3.pt"
    export_packed_int3(source, path)

    manifest = torch.load(path, map_location="cpu", weights_only=True)
    manifest["tensors"]["output.bias"]["packed"][0] ^= 1
    torch.save(manifest, path)

    with pytest.raises(ValueError, match="integrity"):
        PackedInt3Policy.load(path)
