# Crystal-9 F32 ONNX inspection graph

`crystal-9-f32-fixed8-inspection.onnx` is a derived, fixed-shape `[1, 8]` ONNX export of the immutable Crystal-9 F32 reference checkpoint.

It exists for visual inspection in [Netron](https://netron.app/). It is **not** a Crystal-9 runtime, packed artifact, accepted deployment, GGUF, llama.cpp, or Ollama model.

The graph includes the full F32 forward path: token and position embeddings, causal attention, LayerNorm, router/top-2 selection, all nine expert MLP branches, routed merge, and output logits. Input is `token_ids` (`int64`, shape `[1, 8]`); output is `logits` (`float32`, shape `[1, 13]`).

Keep `crystal-9-f32-fixed8-inspection.onnx` beside its required external-weight file, `crystal-9-f32-fixed8-inspection.onnx.data`, when transferring or opening it. `crystal-9-f32-fixed8-inspection-validation.json` records source and both export-file digests, graph inventory, ONNX checker result, and CPU ONNX Runtime comparison. The accepted custom runtimes remain the package artifacts outside this folder.

`crystal-9-f32-fixed8-inspection.onnx.png` is a clipped, indexed-color preview; `crystal-9-f32-fixed8-inspection.onnx.svg` is its zoomable vector counterpart. Both are derived renderings generated from the ONNX file with [Netron 9.2.9](https://netron.app/) by Lutz Roeder, not additional model artifacts.
