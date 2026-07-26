from pathlib import Path

import pytest

import run_gemini_multi_agent1 as ma1
from src.core import evaluate_route
from src.problem_loader import generate_random_problem


def _candidate(
    *,
    problem,
    candidate_id: int,
    route: list[int],
    route_image: str,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "raw_response": "test",
        **evaluate_route(problem, route),
        "artifacts": {"route_image": route_image},
        "timing": {},
    }


def _pending(candidates: list[dict]) -> dict:
    return {
        "iteration": 1,
        "critic": {
            "returned_candidate_count": len(candidates),
            "candidates": candidates,
            "api_call": None,
            "timing": {
                "critic_stage_wall_seconds": 0.1,
            },
        },
        "scorer_attempts": [],
    }


def test_single_valid_candidate_avoids_scorer_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = generate_random_problem(
        num_nodes=4,
        seed=42,
    )
    run_dir = tmp_path / "runs" / "run"
    output = run_dir / "multi_agent1"
    fallback_image = run_dir / "zero_shot" / "images" / "route.png"
    fallback_image.parent.mkdir(parents=True)
    fallback_image.write_bytes(b"image")
    valid_route = [0, 1, 2, 3, 0]
    invalid_route = [0, 1, 1, 3, 0]
    candidates = [
        _candidate(
            problem=problem,
            candidate_id=1,
            route=valid_route,
            route_image="multi_agent1/images/candidate_1.png",
        ),
        _candidate(
            problem=problem,
            candidate_id=2,
            route=invalid_route,
            route_image="multi_agent1/images/candidate_2.png",
        ),
    ]
    monkeypatch.setattr(
        ma1,
        "request_scorer",
        lambda *args, **kwargs: pytest.fail(
            "Tek geçerli adayda scorer API çağrılmamalı."
        ),
    )

    completed, error = ma1._finish_scorer(
        _pending(candidates),
        problem=problem,
        model="test-model",
        output=output,
        run_dir=run_dir,
        fallback_route=valid_route,
        fallback_image=fallback_image,
    )

    assert error is None
    assert completed is not None
    assert (
        completed["scorer"]["selection_mode"]
        == "single_valid_candidate_without_api"
    )
    assert completed["scorer"]["best_candidate_id"] == 1
    assert completed["selected_solution"]["validation"]["is_valid"]


def test_no_valid_candidate_retains_previous_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = generate_random_problem(
        num_nodes=4,
        seed=42,
    )
    run_dir = tmp_path / "runs" / "run"
    output = run_dir / "multi_agent1"
    fallback_image = run_dir / "zero_shot" / "images" / "route.png"
    fallback_image.parent.mkdir(parents=True)
    fallback_image.write_bytes(b"image")
    fallback_route = [0, 1, 2, 3, 0]
    candidates = [
        _candidate(
            problem=problem,
            candidate_id=1,
            route=[0, 1, 1, 3, 0],
            route_image="multi_agent1/images/candidate_1.png",
        )
    ]
    monkeypatch.setattr(
        ma1,
        "request_scorer",
        lambda *args, **kwargs: pytest.fail(
            "Geçerli aday yokken scorer API çağrılmamalı."
        ),
    )

    completed, error = ma1._finish_scorer(
        _pending(candidates),
        problem=problem,
        model="test-model",
        output=output,
        run_dir=run_dir,
        fallback_route=fallback_route,
        fallback_image=fallback_image,
    )

    assert error is None
    assert completed is not None
    assert completed["selected_solution"]["route"] == fallback_route
    assert (
        completed["scorer"]["selection_mode"]
        == "retain_previous_route_no_valid_candidate"
    )


def test_scorer_receives_only_valid_candidate_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = generate_random_problem(
        num_nodes=5,
        seed=42,
    )
    run_dir = tmp_path / "runs" / "run"
    output = run_dir / "multi_agent1"
    fallback_image = run_dir / "zero_shot" / "images" / "route.png"
    fallback_image.parent.mkdir(parents=True)
    fallback_image.write_bytes(b"image")
    routes = [
        [0, 1, 2, 3, 4, 0],
        [0, 1, 1, 3, 4, 0],
        [0, 4, 3, 2, 1, 0],
    ]
    candidates = []
    for candidate_id, route in enumerate(routes, start=1):
        image = (
            run_dir
            / "multi_agent1"
            / "images"
            / f"candidate_{candidate_id}.png"
        )
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        candidates.append(
            _candidate(
                problem=problem,
                candidate_id=candidate_id,
                route=route,
                route_image=image.relative_to(run_dir).as_posix(),
            )
        )

    captured: dict = {}

    class Response:
        text = "image1: 10, image3: 9\nbest route: 1"
        api_call = {
            "api_call_wall_seconds": 0.2,
            "usage": {"total_token_count": 10},
        }

    def fake_request(paths, *, problem, image_ids, model):
        captured["ids"] = image_ids
        captured["problem"] = problem
        return Response()

    monkeypatch.setattr(ma1, "request_scorer", fake_request)

    completed, error = ma1._finish_scorer(
        _pending(candidates),
        problem=problem,
        model="test-model",
        output=output,
        run_dir=run_dir,
        fallback_route=routes[0],
        fallback_image=fallback_image,
    )

    assert error is None
    assert completed is not None
    assert captured["ids"] == [1, 3]
    assert captured["problem"] is problem
    assert completed["scorer"]["excluded_invalid_candidate_ids"] == [2]
    assert completed["scorer"]["best_candidate_id"] == 1


def test_checkpoint_validates_problem_fingerprint() -> None:
    checkpoint = {
        "run_id": "run",
        "model": "model",
        "candidate_count_requested": 7,
        "problem_fingerprint_sha256": "old",
    }
    with pytest.raises(ValueError, match="fingerprint"):
        ma1._validate_checkpoint(
            checkpoint,
            run_id="run",
            model="model",
            candidate_count=7,
            fingerprint="new",
        )
