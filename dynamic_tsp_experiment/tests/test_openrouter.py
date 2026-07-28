import json
from pathlib import Path

import pytest

import src.openrouter as openrouter
from run_openrouter_zero_shot import (
    build_comparison,
    is_ascending_node_id_route,
    output_format_compliant,
)
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


def _manifest() -> dict:
    return {
        "run_id": "random25_run_01",
        "problem": {
            "name": "random_n25_seed42",
            "dimension": 25,
            "depot_id": 0,
            "fingerprint_sha256": "same-problem",
            "reference": {
                "type": "or_tools_heuristic",
                "distance": 17.5,
                "is_proven_optimal": False,
            },
        },
    }


def _write_result(
    run_dir: Path,
    alias: str,
    *,
    valid: bool,
    distance: float,
) -> None:
    path = (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / alias
        / "zero_shot_results.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "problem": {
                    "fingerprint_sha256": "same-problem"
                },
                "validation": {
                    "is_valid": valid,
                    "missing_nodes": [] if valid else [3],
                    "repeated_nodes": [],
                    "unexpected_nodes": [],
                },
                "distance": distance,
                "gap_to_reference_percent": (
                    10.0 if valid else None
                ),
                "api_calls": [
                    {
                        "api_call_wall_seconds": 2.0,
                        "finish_reason": "stop",
                        "usage": {
                            "total_token_count": 100,
                            "cost": 0.0,
                        },
                    }
                ],
                "errors": [],
                "raw_response": "route",
            }
        ),
        encoding="utf-8",
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


def test_output_format_compliance_rejects_extra_commentary() -> None:
    exact = (
        "<<start>>\n"
        "Salesman1: Depot-1-2-Depot\n"
        "<<end>>"
    )
    assert output_format_compliant(exact)
    assert not output_format_compliant(
        exact + "\nI will reconsider the route."
    )


def test_ascending_node_route_is_detected() -> None:
    assert is_ascending_node_id_route(
        [0, 1, 2, 3, 0],
        node_ids=[0, 1, 2, 3],
        depot_id=0,
    )
    assert not is_ascending_node_id_route(
        [0, 2, 1, 3, 0],
        node_ids=[0, 1, 2, 3],
        depot_id=0,
    )


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


def test_comparison_ranks_only_valid_routes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "random25_run_01"
    _write_result(
        run_dir,
        "gemma-4-26b-a4b-it",
        valid=True,
        distance=22.0,
    )
    _write_result(
        run_dir,
        "gemma-4-31b-it",
        valid=True,
        distance=20.0,
    )
    _write_result(
        run_dir,
        "nemotron-3-nano-omni",
        valid=False,
        distance=18.0,
    )

    comparison = build_comparison(
        run_dir=run_dir,
        manifest=_manifest(),
    )

    assert comparison["counts"]["valid_route_count"] == 2
    assert comparison["best_valid_model"]["alias"] == (
        "gemma-4-31b-it"
    )
    assert [
        row["alias"]
        for row in comparison["ranking_by_valid_distance"]
    ] == [
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
    ]


def test_equal_best_distances_are_reported_as_tie(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "random25_run_01"
    _write_result(
        run_dir,
        "nemotron-3-nano-omni",
        valid=True,
        distance=61.0,
    )
    _write_result(
        run_dir,
        "nemotron-nano-12b-v2-vl",
        valid=True,
        distance=61.0,
    )

    comparison = build_comparison(
        run_dir=run_dir,
        manifest=_manifest(),
    )

    assert comparison["best_distance_is_tied"]
    assert len(
        comparison["best_valid_models_at_same_distance"]
    ) == 2


def test_comparison_rejects_different_problem(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "random25_run_01"
    _write_result(
        run_dir,
        "gemma-4-31b-it",
        valid=True,
        distance=20.0,
    )
    path = (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / "gemma-4-31b-it"
        / "zero_shot_results.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["problem"]["fingerprint_sha256"] = "other"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="farklı probleme"):
        build_comparison(
            run_dir=run_dir,
            manifest=_manifest(),
        )
