"""GitHub release bundle coverage for accepted Crystal-9 artifacts."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).parents[1]
STAGE = ROOT / "releases/huggingface-int4-v1"


def load_builder():
    spec = spec_from_file_location("github_release_bundle", ROOT / "scripts/build_github_release_bundle.py")
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_bundle_contains_every_manifest_model_with_verified_digests(tmp_path: Path):
    builder = load_builder()
    result = builder.build_release_bundle(STAGE, tmp_path)

    assets = json.loads(result.assets_manifest.read_text())
    expected = json.loads((STAGE / "release-manifest.json").read_text())
    model_ids = {entry["id"] for entry in [*expected["artifacts"], *expected["research_artifacts"]]}

    assert {entry["id"] for entry in assets["artifacts"]} == model_ids
    assert result.archive.is_file()
    assert (tmp_path / "SHA256SUMS").is_file()
    assert (tmp_path / "release-manifest.json").is_file()
    with zipfile.ZipFile(result.archive) as archive:
        names = set(archive.namelist())
    assert "crystal-9-accepted-artifacts-v1/artifacts/crystal-9-int4-group2-packed-fp16-scales-v1.pt" in names
    assert "crystal-9-accepted-artifacts-v1/release-manifest.json" in names


def test_github_workflow_validates_then_publishes_tagged_release_assets():
    workflow = (ROOT / ".github/workflows/release-official-models.yml").read_text()

    assert "workflow_dispatch:" in workflow
    assert "refs/tags/crystal-9-v" in workflow
    assert "validation/verify_release.py" in workflow
    assert "gh release create" in workflow
    assert "dist/artifacts/*.pt" in workflow
