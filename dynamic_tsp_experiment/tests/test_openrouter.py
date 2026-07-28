from pathlib import Path

import pytest

import src.openrouter as openrouter
from src.openrouter import (
    OPENROUTER_MODELS,
    OpenRouterAPIError,
    OpenRouterCandidatesResult,
    build_multimodal_payload,
    build_route_payload,
    normalize_usage,
    request_candidates,
    resolve_model_alias,
)


def test_registered_models_are_fixed_openrouter_ids() -> None:
    assert set(OPENROUTER_MODELS) == {
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
        "nemotron-3-nano-omni",
        "nemotron-nano-12b-v2-vl",
    }
    assert resolve_model_alias(
        "gemma-4-31b-it"
    ) == "google/gemma-4-31b-it:free"
    assert all(
        model != "openrouter/free"
        for model in OPENROUTER_MODELS.values()
    )


def test_unknown_model_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen"):
        resolve_model_alias("groq-qwen36-repair")


def test_body_level_openrouter_error_is_preserved() -> None:
    with pytest.raises(
        OpenRouterAPIError,
        match="provider overloaded",
    ):
        openrouter._response_texts(
            {
                "error": {
                    "code": 503,
                    "message": "provider overloaded",
                }
            }
        )


def test_payload_uses_same_prompt_and_disables_reasoning() -> None:
    payload = build_route_payload(
        model="provider/model",
        prompt="same prompt",
        image_data_url="data:image/png;base64,abc",
        temperature=0.0,
        max_tokens=1024,
        reasoning_effort="none",
    )
    assert payload["temperature"] == 0.0
    assert payload["reasoning"]["effort"] == "none"
    assert payload["messages"][0]["content"][0]["text"] == (
        "same prompt"
    )
    assert (
        payload["messages"][0]["content"][1]["image_url"][
            "url"
        ]
        == "data:image/png;base64,abc"
    )


def test_multi_agent_payload_supports_native_candidates_and_images() -> None:
    payload = build_multimodal_payload(
        model="provider/model",
        prompt="score routes",
        image_data_urls=[
            "data:image/png;base64,one",
            "data:image/png;base64,two",
        ],
        image_ids=[3, 7],
        temperature=0.7,
        max_tokens=2048,
        reasoning_effort="none",
        candidate_count=7,
    )

    assert payload["n"] == 7
    content = payload["messages"][0]["content"]
    assert content[0]["text"] == "score routes"
    assert content[1]["text"] == "Image 3:"
    assert content[3]["text"] == "Image 7:"
    assert sum(
        item["type"] == "image_url"
        for item in content
    ) == 2


def test_independent_candidate_strategy_returns_requested_count(
    monkeypatch,
) -> None:
    call_number = 0

    def fake_request(*args, **kwargs):
        nonlocal call_number
        call_number += 1
        call = {
            "phase": kwargs["phase"],
            "provider": "openrouter",
            "model": kwargs["model"],
            "response_model": kwargs["model"],
            "routed_provider": "provider",
            "temperature": kwargs["temperature"],
            "reasoning_effort": kwargs["reasoning_effort"],
            "max_tokens": kwargs["max_tokens"],
            "success": True,
            "started_at_utc": f"start-{call_number}",
            "finished_at_utc": f"finish-{call_number}",
            "api_call_wall_seconds": 1.0,
            "input_image_count": 1,
            "input_image_bytes": 10,
            "finish_reason": "stop",
            "usage": {
                "prompt_token_count": 10,
                "candidates_token_count": 2,
                "thoughts_token_count": None,
                "cached_content_token_count": None,
                "total_token_count": 12,
                "cost": 0.0,
            },
        }
        return OpenRouterCandidatesResult(
            texts=[f"route-{call_number}"],
            api_call=call,
            api_calls=[call],
        )

    monkeypatch.setattr(
        openrouter,
        "_request_multimodal",
        fake_request,
    )

    result = request_candidates(
        Path("unused.png"),
        prompt="critic",
        candidate_count=3,
        model="provider/model",
        strategy="independent_calls",
    )

    assert result.texts == [
        "route-1",
        "route-2",
        "route-3",
    ]
    assert len(result.api_calls) == 3
    assert result.api_call["http_request_count"] == 3
    assert result.api_call["returned_candidate_count"] == 3
    assert result.api_call["usage"]["total_token_count"] == 36.0


def test_openrouter_usage_is_normalized() -> None:
    usage = normalize_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cost": 0.001,
            "is_byok": False,
            "prompt_tokens_details": {
                "cached_tokens": 2
            },
            "completion_tokens_details": {
                "reasoning_tokens": 4
            },
        }
    )
    assert usage["prompt_token_count"] == 10
    assert usage["candidates_token_count"] == 20
    assert usage["total_token_count"] == 30
    assert usage["thoughts_token_count"] == 4
    assert usage["cached_content_token_count"] == 2
    assert usage["cost"] == 0.001
