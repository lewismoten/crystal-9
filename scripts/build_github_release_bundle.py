#!/usr/bin/env python3
"""Create checksum-verified GitHub Release assets for Crystal-9."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / "releases/huggingface-int4-v1"
DISTRIBUTION_FILES = (
    "LICENSE",
    "README.md",
    "requirements.txt",
    "crystal9.py",
    "design.json",
    "packed_int2.py",
    "packed_int3.py",
    "packed_int4.py",
    "play_crystal9.py",
    "release-manifest.json",
    "SHA256SUMS",
    "validation/release-acceptance.json",
    "validation/packed-int2-fp8-research-acceptance.json",
    "validation/packed-int3-acceptance.json",
    "validation/packed-int4-acceptance.json",
    "validation/verify_release.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source: Path, stage: Path, package_root: Path) -> None:
    destination = package_root / source.relative_to(stage)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_release_bundle(stage: Path, output: Path) -> SimpleNamespace:
    stage = stage.resolve()
    output = output.resolve()
    manifest = json.loads((stage / "release-manifest.json").read_text())
    release_id = manifest["release_id"]
    artifacts = [*manifest["artifacts"], *manifest["research_artifacts"]]

    package_root = output / release_id
    archive = output / f"{release_id}.zip"
    assets_dir = output / "artifacts"
    assets_manifest = output / "release-assets-manifest.json"
    notes = output / "RELEASE_NOTES.md"
    if package_root.exists():
        shutil.rmtree(package_root)
    if archive.exists():
        archive.unlink()
    shutil.rmtree(assets_dir, ignore_errors=True)
    package_root.mkdir(parents=True)
    assets_dir.mkdir()

    published = []
    for artifact in artifacts:
        source = stage / artifact["path"]
        actual = digest(source)
        if actual != artifact["sha256"]:
            raise ValueError(f"artifact digest mismatch for {artifact['path']}: {actual}")
        if source.stat().st_size != artifact["bytes"]:
            raise ValueError(f"artifact size mismatch for {artifact['path']}")
        copy_file(source, stage, package_root)
        shutil.copy2(source, assets_dir / source.name)
        published.append({
            "id": artifact["id"],
            "path": artifact["path"],
            "sha256": actual,
            "bytes": source.stat().st_size,
        })

    for relative_path in DISTRIBUTION_FILES:
        copy_file(stage / relative_path, stage, package_root)

    assets_manifest.write_text(json.dumps({
        "release_id": release_id,
        "artifacts": published,
    }, indent=2) + "\n")
    shutil.copy2(stage / "release-manifest.json", output / "release-manifest.json")
    shutil.copy2(stage / "SHA256SUMS", output / "SHA256SUMS")
    shutil.copy2(assets_manifest, package_root / assets_manifest.name)
    notes.write_text(
        f"# {release_id}\n\n"
        "Verified Crystal-9 release artifacts. Download an individual `.pt` file "
        "or the complete ZIP, then verify with `SHA256SUMS`.\n"
    )
    shutil.copy2(notes, package_root / notes.name)

    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output))

    return SimpleNamespace(archive=archive, assets_manifest=assets_manifest, notes=notes, artifacts=published)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    result = build_release_bundle(args.stage, args.output)
    print(json.dumps({
        "archive": str(result.archive),
        "assets_manifest": str(result.assets_manifest),
        "artifact_count": len(result.artifacts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
