# Crystal-9 packed INT3 v1

Staged, **not published** release candidate for the complete accepted mixed-layout INT3 policy.

- Artifact: `artifacts/crystal-9-int3-packed-v1.pt`
- Runtime: `packed_int3.PackedInt3Policy`
- Exact acceptance: `0 / 294,778` legal-policy misses
- Integrity: artifact checksum plus the container manifest integrity digest

Verify staged contents from this directory:

```sh
sha256sum -c SHA256SUMS
../../.venv/bin/python -c "from packed_int3 import PackedInt3Policy; print(PackedInt3Policy.load('artifacts/crystal-9-int3-packed-v1.pt').manifest['format'])"
```

This custom Python container is not a Transformers checkpoint, GGUF, llama.cpp, or Ollama model. It packs signed three-bit codes low-bit-first with FP32 dequantization scales. Scalar-group tensors retain one scale per scalar, so this does not claim scale-optimal storage.

Publication requires an approved hosted repository target and external upload approval.
