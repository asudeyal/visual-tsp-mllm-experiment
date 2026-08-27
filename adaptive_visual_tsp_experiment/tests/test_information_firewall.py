from pathlib import Path

from src.prompts import PromptSet


def test_prompt_set_contains_policy_but_no_runtime_numeric_feedback():
    root = Path(__file__).resolve().parents[1]
    prompts = PromptSet(root / "prompts", "v1")
    text = prompts.combined("critic")
    # The policy may name forbidden information to prohibit it, but no runtime
    # objective value or route sequence is interpolated into the prompt.
    assert "Current distance =" not in text
    assert "Gap =" not in text
    assert "Current route:" not in text
    assert "432.10" not in text


def test_repair_prompt_does_not_reveal_missing_node_identity():
    root = Path(__file__).resolve().parents[1]
    prompts = PromptSet(root / "prompts", "v1")
    text = prompts.combined("repair")
    assert "Missing node:" not in text
    assert "Coordinates:" not in text
