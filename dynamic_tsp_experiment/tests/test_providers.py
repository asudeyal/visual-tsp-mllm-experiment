import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from src import gemini
from src.problem_loader import generate_random_problem
from src.providers.gemini_provider import GeminiProvider
from src.request_control import (
    RequestController,
    RetryPolicy,
)

import src.providers.groq_provider as groq_provider_module

from src import openrouter
from src.providers.base import ProviderTextResult
from src.providers.openrouter_provider import (
    OpenRouterProvider,
)

from src.providers.registry import (
    DEFAULT_MODELS,
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
    GroqProvider,
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
        "mistral",
    )

    gemini = create_provider("gemini", None)
    groq = create_provider("groq", None)
    mistral = create_provider("mistral", None)

    assert gemini.provider_id == "gemini"
    assert gemini.default_candidate_strategy == (
        "native_multiple_choices"
    )

    assert groq.provider_id == "groq"
    assert groq.resolved_model == "qwen/qwen3.6-27b"
    assert groq.default_candidate_strategy == (
        "independent_calls"
    )

    assert DEFAULT_MODELS["mistral"] == (
        "mistral-small-latest"
    )
    assert mistral.provider_id == "mistral"
    assert mistral.resolved_model == (
        "mistral-small-latest"
    )
    assert mistral.default_candidate_strategy == (
        "independent_calls"
    )


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

def _zero_delay_request_controller() -> RequestController:
    return RequestController(
        retry_policy=RetryPolicy(
            max_retries=1,
            base_delay_seconds=0.0,
            maximum_delay_seconds=0.0,
        )
    )


def _transient_service_error() -> RuntimeError:
    error = RuntimeError("Service unavailable")
    error.status_code = 503
    return error


def test_gemini_route_uses_request_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_request_route(
        *args,
        **kwargs,
    ) -> gemini.GeminiTextResult:
        nonlocal call_count

        call_count += 1

        if call_count == 1:
            raise _transient_service_error()

        return gemini.GeminiTextResult(
            text="Salesman1: Depot-1-Depot",
            api_call={
                "phase": kwargs["phase"],
                "success": True,
                "api_call_wall_seconds": 0.25,
                "usage": {
                    "total_token_count": 12,
                },
            },
        )

    monkeypatch.setattr(
        gemini,
        "request_route",
        fake_request_route,
    )

    controller = _zero_delay_request_controller()
    provider = GeminiProvider(
        "gemini-test-model"
    )
    provider.configure_request_controller(
        controller
    )

    result = provider.request_route(
        Path("unused.png"),
        prompt="test prompt",
        temperature=0.0,
        phase="route_generation",
    )

    assert call_count == 2

    request_control = result.api_call[
        "request_control"
    ]

    assert request_control["success"] is True
    assert request_control["attempt_count"] == 2
    assert request_control["retry_count"] == 1
    assert (
        request_control["active_wall_seconds"]
        >= 0
    )

    assert result.api_call[
        "provider_timing"
    ]["available"] is False

    summary = controller.summary()

    assert summary["execution_count"] == 1
    assert summary["retry_count"] == 1


def test_gemini_independent_candidates_retry_only_failed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_attempt_count = 0

    def fake_request_candidates(
        *args,
        **kwargs,
    ) -> gemini.GeminiCandidatesResult:
        nonlocal api_attempt_count

        api_attempt_count += 1

        if api_attempt_count == 2:
            raise _transient_service_error()

        return gemini.GeminiCandidatesResult(
            texts=[
                f"candidate-{api_attempt_count}"
            ],
            api_call={
                "phase": (
                    "critic_candidate_generation"
                ),
                "success": True,
                "api_call_wall_seconds": 0.1,
                "usage": {
                    "total_token_count": 10,
                },
            },
        )

    monkeypatch.setattr(
        gemini,
        "request_candidates",
        fake_request_candidates,
    )

    problem = generate_random_problem(
        4,
        seed=42,
    )
    controller = _zero_delay_request_controller()
    provider = GeminiProvider(
        "gemini-test-model"
    )
    provider.configure_request_controller(
        controller
    )

    result = provider.request_candidates(
        Path("unused.png"),
        problem=problem,
        candidate_count=2,
        temperature=0.7,
        strategy="independent_calls",
    )

    # İlk aday bir kez çalışır. İkinci adayın ilk
    # isteği başarısız olur ve yalnız o istek tekrarlanır.
    assert api_attempt_count == 3
    assert len(result.texts) == 2
    assert len(result.api_calls) == 2

    first_request = result.api_calls[0][
        "request_control"
    ]
    second_request = result.api_calls[1][
        "request_control"
    ]

    assert first_request["attempt_count"] == 1
    assert first_request["retry_count"] == 0

    assert second_request["attempt_count"] == 2
    assert second_request["retry_count"] == 1

    assert controller.summary()[
        "execution_count"
    ] == 2

def test_groq_request_control_and_provider_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_attempt_count = 0

    def fake_groq_request(
        *args,
        **kwargs,
    ) -> ProviderTextResult:
        nonlocal api_attempt_count

        api_attempt_count += 1

        if api_attempt_count == 1:
            raise _transient_service_error()

        return ProviderTextResult(
            text="Salesman1: Depot-1-Depot",
            api_call={
                "phase": kwargs["phase"],
                "success": True,
                "api_call_wall_seconds": 0.8,
                "usage": {
                    "total_token_count": 20,
                    "raw": {
                        "queue_time": 0.1,
                        "prompt_time": 0.2,
                        "completion_time": 0.3,
                        "total_time": 0.6,
                    },
                },
            },
        )

    monkeypatch.setattr(
        groq_provider_module,
        "_request",
        fake_groq_request,
    )

    controller = _zero_delay_request_controller()
    provider = GroqProvider(
        "groq-test-model"
    )
    provider.configure_request_controller(
        controller
    )

    result = provider.request_route(
        Path("unused.png"),
        prompt="test prompt",
        temperature=0.0,
        phase="route_generation",
    )

    assert api_attempt_count == 2

    request_control = result.api_call[
        "request_control"
    ]

    assert request_control["attempt_count"] == 2
    assert request_control["retry_count"] == 1

    provider_timing = result.api_call[
        "provider_timing"
    ]

    assert provider_timing["available"] is True
    assert provider_timing[
        "provider_queue_seconds"
    ] == pytest.approx(0.1)
    assert provider_timing[
        "provider_prompt_seconds"
    ] == pytest.approx(0.2)
    assert provider_timing[
        "provider_completion_seconds"
    ] == pytest.approx(0.3)
    assert provider_timing[
        "provider_total_seconds"
    ] == pytest.approx(0.6)
    assert provider_timing[
        "estimated_network_or_client_overhead_seconds"
    ] == pytest.approx(0.2)


def test_openrouter_independent_candidates_retry_only_failed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_attempt_count = 0

    def fake_openrouter_candidates(
        *args,
        **kwargs,
    ) -> openrouter.OpenRouterCandidatesResult:
        nonlocal api_attempt_count

        api_attempt_count += 1

        assert kwargs["candidate_count"] == 1
        assert kwargs["strategy"] == (
            "native_multiple_choices"
        )

        if api_attempt_count == 2:
            raise _transient_service_error()

        call = {
            "phase": "critic_candidate_generation",
            "provider": "openrouter",
            "model": kwargs["model"],
            "success": True,
            "api_call_wall_seconds": 0.2,
            "input_image_count": 1,
            "input_image_bytes": 100,
            "response_model": kwargs["model"],
            "routed_provider": "test-provider",
            "finish_reason": "stop",
            "usage": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        }

        return openrouter.OpenRouterCandidatesResult(
            texts=[
                f"candidate-{api_attempt_count}"
            ],
            api_call=call,
            api_calls=[call],
        )

    monkeypatch.setattr(
        openrouter,
        "request_candidates",
        fake_openrouter_candidates,
    )

    problem = generate_random_problem(
        4,
        seed=42,
    )
    controller = _zero_delay_request_controller()
    provider = OpenRouterProvider(
        "test/vision-model"
    )
    provider.configure_request_controller(
        controller
    )

    result = provider.request_candidates(
        Path("unused.png"),
        problem=problem,
        candidate_count=2,
        temperature=0.7,
        strategy="independent_calls",
    )

    assert api_attempt_count == 3
    assert len(result.texts) == 2
    assert len(result.api_calls) == 2

    first_request = result.api_calls[0][
        "request_control"
    ]
    second_request = result.api_calls[1][
        "request_control"
    ]

    assert first_request["attempt_count"] == 1
    assert first_request["retry_count"] == 0

    assert second_request["attempt_count"] == 2
    assert second_request["retry_count"] == 1

    assert result.api_call[
        "http_request_count"
    ] == 2
    assert result.api_call[
        "returned_candidate_count"
    ] == 2
    assert result.api_call[
        "usage"
    ]["total_token_count"] == pytest.approx(30)