"""Build a checksum-bound distribution for the accepted scalar-scale INT2 research artifact."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


CLASSIFICATION = "staged research artifact; not a deployable INT2 release"
RUNTIME_FILES = (
    "packed_int2_fp8_scale_lzma_fixed_permutation_stream_binary_bitplane_zlib_codes.py",
    "packed_int2_fp8_scale_lzma_binary_bitplane_zlib_codes.py",
    "packed_int2_fp8_scale_lzma_binary_permuted_bitplane_zlib_codes.py",
    "packed_int2.py",
    "crystal9.py",
    "design.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_distribution(
    root: Path,
    artifact: Path,
    report: Path,
    output: Path,
    runtime_files: tuple[str, ...] = RUNTIME_FILES,
) -> dict:
    """Copy an accepted research artifact and its independently checkable runtime closure."""
    root, artifact, report, output = map(Path, (root, artifact, report, output))
    acceptance = json.loads(report.read_text())["acceptance"]
    if acceptance != {"legal_histories": 294778, "policy_misses": 0}:
        raise ValueError("only an exact exhaustive INT2 artifact may be distributed")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    copied = {artifact.name: artifact, "acceptance-report.json": report}
    copied.update({f"runtime/{name}": root / name for name in runtime_files})
    for relative, source in copied.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest = {
        "format": "crystal-9-int2-staged-research-distribution-v1",
        "classification": CLASSIFICATION,
        "artifact": artifact.name,
        "source_acceptance_report": "acceptance-report.json",
        "acceptance": acceptance,
        "runtime_files": [f"runtime/{name}" for name in runtime_files],
        "files": {relative: _sha256(output / relative) for relative in copied},
    }
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Crystal-9 exact scalar-scale INT2 research artifact\n\n"
        "This distribution is an exact behavioral research artifact, not a deployable INT2 release. "
        "It retains one FP8 dequantization scale per scalar. Verify copied files with `sha256sum -c SHA256SUMS`.\n"
    )
    checksum_files = [*copied, "release-manifest.json", "README.md"]
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(output / relative)}  {relative}\n" for relative in checksum_files)
    )
    return manifest
