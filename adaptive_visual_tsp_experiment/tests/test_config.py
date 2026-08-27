from pathlib import Path

import pytest

from src.config import load_config


def test_generic_pilot_config_accepts_cli_run_settings():
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "pilot10_v1.yaml",
        provider_name="gemini",
        model="gemini-3.6-flash",
        seed=7,
    )
    assert config.name == "pilot10_v1"
    assert config.run_label == "p10"
    assert config.iterations == 10
    assert config.seed == 7
    assert config.critic.candidates == 3
    assert config.repair.max_attempts == 2
    assert config.stagnation.window == 5
    assert config.provider.name == "gemini"
    assert config.provider.model == "gemini-3.6-flash"
    assert config.provider.candidate_strategy == "independent_calls"


def test_generic_config_requires_provider_and_model():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="provider"):
        load_config(root / "configs" / "pilot10_v1.yaml")
