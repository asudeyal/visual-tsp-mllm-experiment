import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from src.providers.registry import (
    create_provider,
    model_slug,
    provider_model_root,
    supported_providers,
    zero_shot_result_candidates,
)
from src.providers.groq_provider import (
    GROQ_MULTI_IMAGE_MAX_DIMENSION,
    GROQ_QWEN_MAX_COMPLETION_TOKENS,
    GROQ_QWEN_REASONING_MODEL,
    GROQ_SINGLE_IMAGE_MAX_DIMENSION,
    GROQ_USER_AGENT,
    _data_url,
    _model_request_settings,
    _request_headers,
    _upload_max_dimension,
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


def test_groq_headers_avoid_default_urllib_signature() -> None:
    headers = _request_headers("secret")
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Accept"] == "application/json"
    assert headers["User-Agent"] == GROQ_USER_AGENT
    assert "urllib" not in headers["User-Agent"].lower()


def test_groq_upload_dimension_depends_on_image_count() -> None:
    assert _upload_max_dimension(1) == (
        GROQ_SINGLE_IMAGE_MAX_DIMENSION
    )
    assert _upload_max_dimension(2) == (
        GROQ_MULTI_IMAGE_MAX_DIMENSION
    )
    assert _upload_max_dimension(5) == (
        GROQ_MULTI_IMAGE_MAX_DIMENSION
    )
    with pytest.raises(ValueError, match="pozitif"):
        _upload_max_dimension(0)


def test_qwen_disables_explicit_reasoning_and_caps_output() -> None:
    settings, max_tokens = _model_request_settings(
        GROQ_QWEN_REASONING_MODEL,
        max_tokens=4096,
    )
    assert settings == {"reasoning_effort": "none"}
    assert max_tokens == GROQ_QWEN_MAX_COMPLETION_TOKENS

    other_settings, other_max_tokens = _model_request_settings(
        "another/vision-model",
        max_tokens=4096,
    )
    assert other_settings == {}
    assert other_max_tokens == 4096


def test_qwen_inference_settings_are_visible_in_metadata() -> None:
    provider = create_provider("groq", GROQ_QWEN_REASONING_MODEL)
    assert provider.model_metadata["inference_settings"] == {
        "reasoning_effort": "none",
        "route_max_completion_tokens": 1024,
        "scorer_max_completion_tokens": 1024,
    }


def test_groq_data_url_resizes_only_upload_copy(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    original_image = Image.new("RGB", (1800, 1200), "white")
    original_image.save(image_path, format="PNG")
    original_bytes = image_path.read_bytes()

    data_url, metadata = _data_url(
        image_path,
        max_dimension=768,
    )

    assert data_url.startswith("data:image/png;base64,")
    encoded = data_url.split(",", maxsplit=1)[1]
    uploaded = base64.b64decode(encoded)
    with Image.open(BytesIO(uploaded)) as uploaded_image:
        assert uploaded_image.size == (768, 512)
    assert metadata == {
        "original_bytes": len(original_bytes),
        "uploaded_bytes": len(uploaded),
        "original_width": 1800,
        "original_height": 1200,
        "uploaded_width": 768,
        "uploaded_height": 512,
        "resized_for_upload": True,
    }
    assert image_path.read_bytes() == original_bytes
