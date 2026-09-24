# Crystal-9

A clean-room experiment toward the smallest truthful local MoE tic-tac-toe policy model.

## Locked initial design

- 32-wide hidden state, 8 attention heads, 2 KV heads
- 64-wide dense intermediate and 32-wide routed experts
- 9 routed experts, top-2 routing
- eight-token raw move-history context
- explicit thirteen-token game vocabulary: `<pad>`, `<bos>`, `<eos>`, `!`, and `a`–`i`

## Precision policy

FP32 is the source baseline. F16, Q6_K, Q4_K_M, Q3_K, Q2_K, and experimental 1-bit-family formats are candidates, not promises. A candidate is publishable only after the exact target runtime passes the complete legal-policy and invalid-input gates.

The custom vocabulary is intentionally separate from the prior Ollama-compatible byte-BPE vocabulary. Crystal-9 must prove token-ID parity and target-runtime behavior before claiming Ollama compatibility. For a GGUF or standard llama.cpp/Ollama-compatible model today, use [Palace-9](https://huggingface.co/lewismoten/palace-9).

## First gate

```sh
python3 -m pytest tests/ -q
```
