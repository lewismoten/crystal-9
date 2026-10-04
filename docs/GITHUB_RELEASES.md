# Publishing official Crystal-9 models on GitHub

The GitHub workflow at `.github/workflows/release-official-models.yml` publishes only the checksum-verified artifacts named by `releases/huggingface-int4-v1/release-manifest.json`.

Each GitHub Release receives:

- five individual model `.pt` downloads;
- `crystal-9-accepted-artifacts-v1.zip`, containing the models, custom runtime, validation reports, manifest, checksums, and local demo;
- `SHA256SUMS`, `release-manifest.json`, and a concise release-assets manifest.

Before it publishes, GitHub Actions installs CPU-only PyTorch and NumPy, then runs the exhaustive `validation/verify_release.py` gate. A failed policy or integrity check prevents the release.

## Release from GitHub

After this Forgejo commit is mirrored to GitHub, create and push an annotated tag:

```sh
git tag -a v1.0.0 -m "Crystal-9 official models v1.0.0"
git push github v1.0.0
```

Pushing a semantic version tag matching `v*` starts the workflow. It creates a GitHub Release with that tag. Re-running the workflow for an existing tag refreshes its assets without changing the tag.

Alternatively, use **Actions → Release official Crystal-9 models → Run workflow** and enter a new tag such as `v1.0.0`.

The release is intentionally limited to the accepted F32, INT4, INT3, and clearly marked INT2 research artifacts. It does not publish unverified training checkpoints, browser projections, or inspection-only ONNX files.
