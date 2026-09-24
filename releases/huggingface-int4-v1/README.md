---
library_name: crystal-9-custom
license: apache-2.0
pipeline_tag: text-classification
tags:
  - tic-tac-toe
  - mixture-of-experts
  - moe
  - quantization
  - int4
  - local-inference
---

# Crystal-9 accepted artifacts v1

Crystal-9 is a clean-room, local 3×3 tic-tac-toe move-policy experiment. It is a small sparse Mixture-of-Experts policy model: shared token and position embeddings, causal multi-head attention, LayerNorm, a router, nine two-layer experts, and top-2 routing.

This package contains four accepted Crystal-9 artifacts: the immutable F32 reference, two independently validated packed-INT4 deployments, and a complete mixed-layout packed-INT3 deployment. It is **not a Transformers checkpoint, GGUF, llama.cpp, or Ollama model**; use the included custom Python runtime.

> **Need a GGUF or standard llama.cpp/Ollama compatibility?** Try [Palace-9](https://huggingface.co/lewismoten/palace-9), the earlier compatibility-focused Crystal-9 predecessor. Its published GGUF artifacts are the appropriate choice for those runtimes.

## Accepted artifacts

Every listed artifact passed the exhaustive gate: **0 policy misses across 294,778 legal histories**.

### F32 reference

`artifacts/artifacts-fp32.pt` — immutable F32 source and evaluation baseline.
**116,365 bytes** · SHA-256: `e5e3aa5eee…3537c34312b9399c`

### INT4 with FP32 scales

`artifacts/crystal-9-int4-group2-packed-v1.pt` — original accepted deployment: packed signed INT4 codes with FP32 dequantization scales.
**46,547 bytes** · SHA-256: `10fb96a66aaf…da12434a2c722b9f9`

### INT4 with FP16 scales

`artifacts/crystal-9-int4-group2-packed-fp16-scales-v1.pt` — accepted scale-compressed deployment: the same signed INT4-code layout with all 787 dequantization scales stored as FP16.
**46,299 bytes** · SHA-256: `63eee663a143…b2f5e329dacb1392eb`

### INT3 with mixed FP32 scales

`artifacts/crystal-9-int3-packed-v1.pt` — complete accepted deployment: packed signed INT3 codes with the recorded mixed FP32-scale layouts.
**55,489 bytes** · SHA-256: `2bc68216b05d…47747b2122174dc8574fc`

Full artifact digests, acceptance evidence, and source provenance are in `release-manifest.json`; `SHA256SUMS` verifies every shipped file.

The FP16-scale artifact is **not a full-FP16 model**. Its model codes remain INT4; only the explicit dequantization scales use FP16. It is 248 bytes (0.53%) smaller than the FP32-scale packed artifact.

A full-FP16 Crystal-9 model has not been created or accepted. The INT3 artifact is complete and accepted, but its mixed groups retain FP32 scales—especially scalar groups—so it is not claimed to be scale-storage-optimal. Its internal manifest is independently protected by SHA-256 `5a27545c39fa2b327e25f8f62c4a16f4a820643cb3354ac36b4e97f2c115c58a`.

## Why Crystal-9 followed Palace-9

[Palace-9](https://huggingface.co/lewismoten/palace-9) was the earlier compatibility-focused experiment: a `Qwen2MoeForCausalLM` model shaped for Transformers, llama.cpp, and Ollama chat tooling. That required a general-purpose byte-BPE vocabulary, a 16-token context, and architecture/configuration conventions intended for another model family. Those constraints were useful for proving compatibility, but they were not the most compact fit for a deterministic 3×3 move-history policy.

Crystal-9 uses the same deterministic tic-tac-toe training and evaluation data, but was redesigned around the task: a 32-wide hidden state, nine top-2 routed experts, an eight-move context, a 13-token game vocabulary, and exact packed-INT4 inference. It intentionally uses a custom local runtime rather than a llama.cpp/Ollama-compatible container.

| Comparable artifact | Palace-9 | Crystal-9 | Difference |
|---|---:|---:|---:|
| F32 source checkpoint | `model.safetensors` — 2,598,856 B | F32 reference — 116,365 B | Crystal-9 is 22.33× smaller (95.52% reduction) |
| Compact deployment | Q4_K_M GGUF — 1,065,216 B | INT4 + FP32 scales — 46,547 B | Crystal-9 is 22.88× smaller (95.63% reduction) |
| Compact deployment | Q4_K_M GGUF — 1,065,216 B | INT4 + FP16 scales — 46,299 B | Crystal-9 is 23.01× smaller (95.65% reduction) |

The formats are not interchangeable. Palace-9's listed artifacts are Qwen2-MoE/Transformers or GGUF compatibility artifacts; Crystal-9's artifacts are a PyTorch F32 state dictionary and custom packed-INT4 containers. These comparisons document task-specific design tradeoffs, not loader compatibility.

The exact comparison inputs and byte counts are in `validation/palace-9-comparison.json`.

## Tensor workflow

![Crystal-9 full-parameter INT4 QAT tensor workflow](assets/int4-full-qat-tensor-workflow.png)

This is a decoded, non-reconstructable inspection of the full-parameter INT4 QAT checkpoint. It documents the tensor layout and QAT workflow; it is not a release label, model container, or substitute for either packed deployment artifact.

## Input and output contract

Pass a raw history of board-square letters `a` through `i` in play order, with at most eight moves. The policy returns one square letter for a legal next move, or `!` when the history is invalid or terminal.

The internal vocabulary is `<pad>`, `<bos>`, `<eos>`, `!`, and `a`–`i`. Only `a`–`i` are public input symbols.

## Run locally

Use a current PyTorch installation. From this repository root:

```bash
python3 - <<'PY'
from crystal9 import GameTokenizer
from packed_int4 import PackedInt4Policy

runtime = PackedInt4Policy.load("artifacts/crystal-9-int4-group2-packed-fp16-scales-v1.pt").eval()
tokenizer = GameTokenizer.from_design_file("design.json")
print(runtime.predict("ae", tokenizer))
PY
```

To load the accepted INT3 artifact instead, change the two runtime lines to:

```python
from packed_int3 import PackedInt3Policy
runtime = PackedInt3Policy.load("artifacts/crystal-9-int3-packed-v1.pt").eval()
```

To play the included terminal demo as X against Crystal-9 (O):

```bash
python3 -m pip install -r requirements.txt
python3 play_crystal9.py
```

The demo uses the accepted smaller INT4 artifact with FP16 dequantization scales. Enter one unoccupied `a`–`i` square per turn; the board prints as three rows containing `.`, `x`, and `o`, and the game stops at a win or draw. This is a local custom-runtime demo; it is not hosted inference.

To evaluate all four staged artifacts on the exhaustive legal-history and invalid-input gates:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 validation/verify_release.py
```

## Integrity and provenance

`release-manifest.json` names every accepted artifact, its role, exact byte size, digest, and acceptance boundary. `validation/release-acceptance.json` is generated by the self-contained validation script against the staged copies.

Verify every distributable file (except `SHA256SUMS` itself):

```bash
sha256sum -c SHA256SUMS
```

`SHA256SUMS` detects changes after generation. It is not an authenticated signature; compare artifact digests in `release-manifest.json` with a trusted release reference.

Browser projection files are separate derivative representations and are not required by this Python runtime. The tensor workflow image is a decoded visualization, not a byte container.

## Acknowledgments

CRYSTAL-9 was designed and directed by Lewis Moten. Its code and documentation were developed with assistance from GPT-5.6-terra Med, accessed through Hermes and using Honcho for context and project-memory support. Lewis Moten remains the project designer, maintainer, and publisher.

## License

This staged package is licensed under [Apache License 2.0](LICENSE). Hosted publication still requires repository-owner approval.
