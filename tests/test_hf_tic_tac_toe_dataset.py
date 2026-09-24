"""Dataset generator contract for the standalone Hugging Face corpus."""

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "hf-tic-tac-toe" / "generate_dataset.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("hf_tic_tac_toe_generator", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_statistics_cover_all_ternary_boards_and_symmetry_classes():
    generator = load_generator()
    stats = generator.compute_statistics()

    assert stats == {
        "all_board_states": 19683,
        "reachable_board_states": 5478,
        "invalid_board_states": 14205,
        "all_board_states_rotation_canonical": 4995,
        "all_board_states_rotation_and_reflection_canonical": 2862,
        "reachable_board_states_rotation_canonical": 1383,
        "reachable_board_states_rotation_and_reflection_canonical": 765,
        "nonterminal_legal_histories": 294778,
    }


def test_generated_rows_expose_history_board_text_and_fixed_tie_target(tmp_path):
    generator = load_generator()
    output = tmp_path / "data"
    manifest = generator.generate(output)

    first = (output / "nonterminal-legal-histories.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert first == '{"history":"","next_player":"X","board":".../.../...","board_text":". . .\\n. . .\\n. . .","optimal_move":"e","optimal_moves":["e","b","d","f","h","c","g","i","a"]}'
    assert manifest["counts"]["nonterminal_legal_histories"] == 294778
    assert manifest["files"]["nonterminal-legal-histories.jsonl"]["rows"] == 294778
