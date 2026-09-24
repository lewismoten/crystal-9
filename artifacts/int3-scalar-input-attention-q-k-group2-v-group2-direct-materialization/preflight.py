import json
import sys
from pathlib import Path

import torch

root = Path("/home/ai/crystal-9")
sys.path.insert(0, str(root))
import crystal9
import training

source = root / "artifacts/int3-scalar-input-attention-q-k-group2-qat-200-seed20260952-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2.pt"
out = Path(__file__).parent
layout = "mixed-int3-scalar-input-attention-q-k-group2-v-group2"
tokenizer = crystal9.GameTokenizer.from_design_file(root / "design.json")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
histories = [history for history in training.legal_histories() if training.optimal_move(history) != "!"]
model = training.load_reference_model(source, tokenizer.vocab_size, device).eval()
materialized = crystal9.materialize_mixed_int3_scalar_input_attention_q_k_group2_v_group2(model).to(device).eval()
fake_misses = materialized_misses = 0
with torch.no_grad():
    for start in range(0, len(histories), 4096):
        batch = histories[start:start + 4096]
        inputs = torch.tensor([training.padded(tokenizer, history) for history in batch], device=device)
        fake = model.forward_mixed_int3_scalar_input_attention_q_k_group2_v_group2(inputs).argmax(dim=-1).tolist()
        actual = materialized(inputs).argmax(dim=-1).tolist()
        fake_misses += sum(tokenizer.decode_id(token_id) != training.optimal_move(history) for token_id, history in zip(fake, batch))
        materialized_misses += sum(tokenizer.decode_id(token_id) != training.optimal_move(history) for token_id, history in zip(actual, batch))
report = {
    "layout": layout,
    "source_checkpoint": str(source.relative_to(root)),
    "trainable_tensors": [],
    "quantization": {"V": {"bits": 3, "group_size": 2, "scale_type": "float32", "scale_count": 512, "storage_efficient": False}},
    "acceptance": {"legal_histories": len(histories), "fake_qat_policy_misses": fake_misses, "materialized_policy_misses": materialized_misses},
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
