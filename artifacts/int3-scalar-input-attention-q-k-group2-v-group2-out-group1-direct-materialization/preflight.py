import json
import sys
from pathlib import Path

import torch

root = Path("/home/ai/crystal-9")
sys.path.insert(0, str(root))
from crystal9 import (
    GameTokenizer,
    materialize_mixed_int3_scalar_input_attention_q_k_group2_v_group2_out_group1,
)
from training import legal_histories, load_reference_model, optimal_move, padded

source = root / "artifacts/int3-scalar-input-attention-q-k-group2-v-group2-qat-200-seed20260954-lr1e-4/artifacts-qat-mixed-int3-scalar-input-attention-q-k-group2-v-group2.pt"
out = Path(__file__).parent
layout = "mixed-int3-scalar-input-attention-q-k-group2-v-group2-out-group1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = GameTokenizer.from_design_file(root / "design.json")
model = load_reference_model(source, tokenizer.vocab_size, device).eval()
materialized = materialize_mixed_int3_scalar_input_attention_q_k_group2_v_group2_out_group1(model).eval()
histories = [history for history in legal_histories() if optimal_move(history) != "!"]
fake_misses = materialized_misses = 0
with torch.no_grad():
    for start in range(0, len(histories), 4096):
        batch = histories[start:start + 4096]
        inputs = torch.tensor([padded(tokenizer, history) for history in batch], device=device)
        fake = model.forward_mixed_int3_scalar_input_attention_q_k_group2_v_group2_out_group1(inputs).argmax(dim=-1).tolist()
        actual = materialized(inputs).argmax(dim=-1).tolist()
        fake_misses += sum(tokenizer.decode_id(token_id) != optimal_move(history) for token_id, history in zip(fake, batch))
        materialized_misses += sum(tokenizer.decode_id(token_id) != optimal_move(history) for token_id, history in zip(actual, batch))
report = {
    "layout": layout,
    "source_checkpoint": str(source),
    "trainable_tensors": [],
    "quantization": {
        "attention.out_proj.weight": {
            "bits": 3,
            "group_size": 1,
            "scale_type": "float32",
            "scale_count": model.attention.out_proj.weight.numel(),
            "storage_efficient": False,
        }
    },
    "acceptance": {
        "legal_histories": len(histories),
        "fake_qat_policy_misses": fake_misses,
        "materialized_policy_misses": materialized_misses,
    },
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
