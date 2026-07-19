import json
import sys
from types import SimpleNamespace

import pytest

import run_gemini_multi_agent1 as ma1
from run_gemini_multi_agent1 import better_valid_solution, main
from src.llm_routes import (
    GeminiTextResult,
    ScorerParseError,
    parse_scorer_response,
    request_gemini_route_candidates,
    request_gemini_scorer,
    scorer_prompt,
)


def test_parse_scorer_response_reads_scores_and_best_candidate() -> None:
    response = """<<image1: 3, image2: 4.5, image3: 2>>
<<the best route: 2>>"""
    scores, best = parse_scorer_response(response, expected_image_ids=[1, 2, 3])

    assert scores == {1: 3.0, 2: 4.5, 3: 2.0}
    assert best == 2


def test_parse_scorer_response_accepts_spacing_case_and_decimal_comma() -> None:
    response = """<< IMAGE 1 : 3,5 , image 2: 4 >>
<<The Best Route: image 2>>"""
    scores, best = parse_scorer_response(response, expected_image_ids=[1, 2])

    assert scores == {1: 3.5, 2: 4.0}
    assert best == 2


def test_parse_scorer_response_accepts_plain_equals_format() -> None:
    response = """Image 1 score = 3
Image 2 score = 5
Image 3 score = 4
Best image ID = 2"""
    scores, best = parse_scorer_response(response, expected_image_ids=[1, 2, 3])

    assert scores == {1: 3.0, 2: 5.0, 3: 4.0}
    assert best == 2


def test_parse_scorer_response_accepts_json_like_format() -> None:
    response = '{"image1": 4, "image2": 5, "best_route": 2}'
    scores, best = parse_scorer_response(response, expected_image_ids=[1, 2])

    assert scores == {1: 4.0, 2: 5.0}
    assert best == 2


def test_parse_scorer_response_rejects_missing_expected_score() -> None:
    response = "<<image1: 4, image2: 3>>\n<<the best route: 1>>"

    with pytest.raises(ScorerParseError, match="Eksik"):
        parse_scorer_response(response, expected_image_ids=[1, 2, 3])


def test_parse_scorer_response_infers_best_when_scores_are_complete() -> None:
    response = "<<image1: 10, image2: 10, image3: 9>>"

    scores, best = parse_scorer_response(response, expected_image_ids=[1, 2, 3])

    assert scores == {1: 10.0, 2: 10.0, 3: 9.0}
    assert best == 1


def test_parse_scorer_response_rejects_unscored_best_candidate() -> None:
    response = "<<image1: 4, image2: 3>>\n<<the best route: 3>>"

    with pytest.raises(ScorerParseError, match="bir skor bulunmuyor"):
        parse_scorer_response(response)


def test_scorer_prompt_is_dynamic_and_visual_only() -> None:
    prompt = scorer_prompt([1, 2, 3])

    assert "image1: score, image2: score, image3: score" in prompt
    assert "Complete Node Coverage" in prompt
    assert "Do not calculate or request coordinates or distances" in prompt


def test_better_valid_solution_keeps_only_shorter_valid_candidate() -> None:
    current = {
        "distance": 10.0,
        "validation": {"is_valid": True},
    }
    invalid_shorter = {
        "distance": 9.0,
        "validation": {"is_valid": False},
    }
    valid_longer = {
        "distance": 11.0,
        "validation": {"is_valid": True},
    }
    valid_shorter = {
        "distance": 8.0,
        "validation": {"is_valid": True},
    }

    assert better_valid_solution(current, invalid_shorter) is current
    assert better_valid_solution(current, valid_longer) is current
    assert better_valid_solution(current, valid_shorter) is valid_shorter


def test_validate_only_checks_plan_without_api_key(
    tmp_path, monkeypatch, capsys
) -> None:
    baseline_path = tmp_path / "baseline.json"
    zero_shot_path = tmp_path / "zero_shot.json"
    output_dir = tmp_path / "output"
    baseline_path.write_text(
        json.dumps(
            {
                "locations": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                "solutions": {
                    "or_tools": {"distance": 4.0},
                    "exact": {"distance": 4.0},
                },
            }
        ),
        encoding="utf-8",
    )
    zero_shot_path.write_text(
        json.dumps({"route": [0, 1, 2, 3, 0]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gemini_multi_agent1.py",
            "--baseline",
            str(baseline_path),
            "--zero-shot",
            str(zero_shot_path),
            "--output-dir",
            str(output_dir),
            "--iterations",
            "3",
            "--validate-only",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "API çağrısı yapılmadı" in output
    assert "Tahmini kalan Gemini isteği: 6" in output
    assert (
        output_dir
        / "runs"
        / "default"
        / "zero_shot"
        / "images"
        / "gemini_zero_shot_route.png"
    ).exists()


def test_candidate_request_uses_one_call_with_requested_candidate_count(
    tmp_path, monkeypatch
) -> None:
    from google import genai

    image_path = tmp_path / "route.png"
    image_path.write_bytes(b"not-a-real-png-but-valid-inline-bytes")
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text=f"candidate {index}")]
                        )
                    )
                    for index in range(1, 4)
                ]
            )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key: SimpleNamespace(models=FakeModels()),
    )

    responses = request_gemini_route_candidates(
        image_path,
        prompt="critic",
        candidate_count=3,
    )

    assert responses == ["candidate 1", "candidate 2", "candidate 3"]
    assert captured["config"].candidate_count == 3
    assert captured["config"].temperature == 0.7


def test_candidate_request_preserves_partial_nonempty_response(
    tmp_path, monkeypatch
) -> None:
    from google import genai

    image_path = tmp_path / "route.png"
    image_path.write_bytes(b"inline-image-bytes")

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text="only candidate")]
                        )
                    )
                ]
            )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key: SimpleNamespace(models=FakeModels()),
    )

    responses = request_gemini_route_candidates(
        image_path,
        prompt="critic",
        candidate_count=3,
    )

    assert responses == ["only candidate"]


def test_scorer_request_sends_all_images_in_one_call(tmp_path, monkeypatch) -> None:
    from google import genai

    image_paths = [tmp_path / "candidate1.png", tmp_path / "candidate2.png"]
    for image_path in image_paths:
        image_path.write_bytes(b"inline-image-bytes")
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                text="<<image1: 3, image2: 4>>\n<<the best route: 2>>"
            )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key: SimpleNamespace(models=FakeModels()),
    )

    response = request_gemini_scorer(image_paths, image_ids=[1, 2])

    assert "the best route: 2" in response
    assert len(captured["contents"]) == 5
    assert captured["contents"][1] == "Image 1:"
    assert captured["contents"][3] == "Image 2:"
    assert captured["config"].temperature == 0.0
    assert captured["config"].max_output_tokens == 4000
    assert captured["config"].thinking_config.thinking_budget == 512


def _pending_scorer_fixture(image_path) -> dict:
    evaluation = {
        "route": [0, 1, 2, 3, 0],
        "validation": {"is_valid": True},
        "legal_node_ids": True,
        "distance": 4.0,
        "gap_to_or_tools_percent": 0.0,
        "gap_to_exact_percent": 0.0,
    }
    return {
        "iteration": 2,
        "critic": {
            "candidates": [
                {
                    "candidate_id": 1,
                    "image": str(image_path),
                    **evaluation,
                }
            ]
        },
    }


def test_failed_scorer_response_is_preserved_in_pending_checkpoint(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "candidate.png"
    image_path.write_bytes(b"image")
    pending = _pending_scorer_fixture(image_path)
    monkeypatch.setattr(
        ma1,
        "request_gemini_scorer_detailed",
        lambda *args, **kwargs: GeminiTextResult(
            text="bad format",
            api_call={
                "success": True,
                "api_call_wall_seconds": 0.01,
                "usage": {"total_token_count": 10},
            },
        ),
    )

    with pytest.raises(ScorerParseError):
        ma1.finalize_pending_iteration(
            pending=pending,
            locations=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            args=SimpleNamespace(model="test", output_dir=tmp_path),
        )

    assert pending["scorer_attempts"][0]["raw_response"] == "bad format"
    assert pending["scorer_attempts"][0]["parse_error_type"] == "ScorerParseError"


def test_parseable_stored_scorer_response_avoids_new_api_call(
    tmp_path, monkeypatch
) -> None:
    image_path = tmp_path / "candidate.png"
    image_path.write_bytes(b"image")
    pending = _pending_scorer_fixture(image_path)
    pending["scorer_attempts"] = [
        {
            "raw_response": (
                "Image 1 score = 5\n"
                "Best image ID = 1"
            )
        }
    ]
    monkeypatch.setattr(
        ma1,
        "request_gemini_scorer_detailed",
        lambda *args, **kwargs: pytest.fail("API yeniden çağrılmamalı"),
    )
    monkeypatch.setattr(ma1, "plot_evaluation", lambda *args, **kwargs: None)

    completed, selected = ma1.finalize_pending_iteration(
        pending=pending,
        locations=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        args=SimpleNamespace(model="test", output_dir=tmp_path),
    )

    assert selected["candidate_id"] == 1
    assert completed["scorer"]["scores"] == {"1": 5.0}
