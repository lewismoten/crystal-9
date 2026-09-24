from pathlib import Path


def test_int3_release_manifest_validates_packed_artifact_provenance():
    from tools.verify_int3_release import validate_release

    root = Path(__file__).resolve().parents[1]
    report = validate_release(root / "releases" / "huggingface-int3-v1")

    artifact = report["artifacts"][0]
    assert report["release_status"] == "staged-not-published"
    assert artifact["id"] == "packed-int3-v1"
    assert artifact["acceptance"] == {"legal_histories": 294778, "policy_misses": 0}
    assert artifact["source_checkpoint"].endswith("artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt")
