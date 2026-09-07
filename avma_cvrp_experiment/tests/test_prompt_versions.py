from __future__ import annotations

from pathlib import Path

import pytest

from src.prompts import PromptSet


ROLES = ("initializer", "critic", "scorer", "repair", "hybrid", "diversity")


@pytest.mark.parametrize(
    "version",
    ["cvrp_capacity_v1", "cvrp_capacity_v2", "cvrp_capacity_v3"],
)
def test_all_prompt_versions_are_available(version: str) -> None:
    root = Path(__file__).resolve().parents[1] / "prompts"
    prompts = PromptSet(root, version)

    assert prompts.version == version
    assert prompts.common.strip()

    for role in ROLES:
        assert prompts.role(role).strip()

    hashes = prompts.hashes()
    assert "common_policy.txt" in hashes
    for role in ROLES:
        assert f"{role}.txt" in hashes
