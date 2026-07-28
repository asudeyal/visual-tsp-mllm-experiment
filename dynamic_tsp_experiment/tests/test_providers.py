from pathlib import Path

import pytest

from src.providers.registry import (
    create_provider,
    model_slug,
    provider_model_root,
    supported_providers,
    zero_shot_result_candidates,
)


def test_supported_providers_and_defaults() -> None:
    assert supported_providers() == (
        "gemini",
        "openrouter",
        "groq",
    )
    gemini = create_provider("gemini", None)
    groq = create_provider("groq", None)
    assert gemini.provider_id == "gemini"
    assert gemini.default_candidate_strategy == (
        "native_multiple_choices"
    )
    assert groq.provider_id == "groq"
    assert groq.resolved_model == "qwen/qwen3.6-27b"
    assert groq.default_candidate_strategy == "independent_calls"


def test_openrouter_requires_model() -> None:
    with pytest.raises(ValueError, match="--model"):
        create_provider("openrouter", None)


def test_provider_output_path_is_isolated_by_provider_and_model() -> None:
    run_dir = Path("output") / "runs" / "run1"
    assert provider_model_root(
        run_dir,
        "groq",
        "qwen/qwen3.6-27b",
    ) == (
        run_dir
        / "providers"
        / "groq"
        / "qwen-qwen3.6-27b"
    )
    assert model_slug("google/gemma-4-31b-it:free") == (
        "google-gemma-4-31b-it-free"
    )


def test_groq_candidate_count_respects_five_image_limit() -> None:
    provider = create_provider("groq", None)
    provider.validate_candidate_count(5)
    with pytest.raises(ValueError, match="en fazla 5"):
        provider.validate_candidate_count(6)


def test_zero_shot_candidates_include_compatible_legacy_paths() -> None:
    run_dir = Path("output") / "runs" / "run1"
    gemini = zero_shot_result_candidates(
        run_dir,
        "gemini",
        "gemini-2.5-flash",
    )
    assert gemini[-1] == (
        run_dir / "zero_shot" / "zero_shot_results.json"
    )
    openrouter = zero_shot_result_candidates(
        run_dir,
        "openrouter",
        "nemotron-3-nano-omni",
    )
    assert openrouter[-1] == (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / "nemotron-3-nano-omni"
        / "zero_shot_results.json"
    )
    groq = zero_shot_result_candidates(
        run_dir,
        "groq",
        "qwen/qwen3.6-27b",
    )
    assert len(groq) == 1
