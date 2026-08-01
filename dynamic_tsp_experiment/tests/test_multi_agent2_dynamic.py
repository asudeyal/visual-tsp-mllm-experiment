from pathlib import Path

import pytest

import run_multi_agent2 as ma2
from src.solution_tracking import (
    EarlyStopPolicy,
    SolutionProgressTracker,
)


def _evaluation(
    *,
    distance: float,
    valid: bool,
) -> dict:
    return {
        "route": [0, 1, 0],
        "validation": {"is_valid": valid},
        "legal_node_ids": True,
        "distance": distance,
        "reference_distance": 10.0,
        "gap_to_reference_percent": (
            100.0 * (distance - 10.0) / 10.0
            if valid
            else None
        ),
    }


def test_best_solution_ignores_shorter_invalid_route() -> None:
    initializer = {
        "source": "zero_shot",
        "iteration": 0,
        **_evaluation(distance=15.0, valid=True),
    }
    iterations = [
        {
            "iteration": 1,
            **_evaluation(distance=8.0, valid=False),
        },
        {
            "iteration": 2,
            **_evaluation(distance=12.0, valid=True),
        },
    ]

    best = ma2._best(initializer, iterations)

    assert best is not None
    assert best["iteration"] == 2
    assert best["distance"] == 12.0


def test_checkpoint_requires_same_problem_fingerprint() -> None:
    checkpoint = {
        "run_id": "run1",
        "model": "model1",
        "problem_fingerprint_sha256": "first",
    }

    with pytest.raises(
        ValueError,
        match="fingerprint",
    ):
        ma2._validate_checkpoint(
            checkpoint,
            run_id="run1",
            model="model1",
            fingerprint="second",
        )


def test_run_artifact_cannot_escape_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(
        ValueError,
        match="dışındaki artifact",
    ):
        ma2._resolve_run_artifact(
            run_dir,
            "../outside.png",
        )


def test_request_timing_separates_wait_types() -> None:
    timing = ma2._request_timing(
        {
            "api_call_wall_seconds": 7.0,
            "request_control": {
                "active_wall_seconds": 5.0,
                "total_wall_seconds": 8.0,
                "waits": {
                    "deliberate_delay_seconds": 2.0,
                    "rate_limit_backoff_seconds": 1.0,
                    "controlled_wait_seconds": 3.0,
                },
            },
        }
    )

    assert timing == {
        "api_active_wall_seconds": 5.0,
        "deliberate_delay_seconds": 2.0,
        "rate_limit_backoff_seconds": 1.0,
        "controlled_wait_seconds": 3.0,
        "api_request_total_wall_seconds": 8.0,
    }


def test_resume_replay_reconstructs_system_gbest() -> None:
    tracker = SolutionProgressTracker(
        provider="gemini",
        reference_distance=10.0,
        reference_type="tsplib_known_optimum",
        reference_is_proven_optimal=True,
        early_stop_policy=EarlyStopPolicy(
            threshold_percent=1.0,
        ),
    )
    initializer = {
        "source": "zero_shot",
        "iteration": 0,
        **_evaluation(distance=15.0, valid=True),
    }
    iterations = [
        {
            "iteration": 1,
            **_evaluation(distance=12.0, valid=True),
        },
        {
            "iteration": 2,
            **_evaluation(distance=8.0, valid=False),
        },
    ]

    ma2._replay_solution_progress(
        tracker,
        initializer=initializer,
        iterations=iterations,
    )

    assert tracker.system_gbest is not None
    assert tracker.system_gbest.distance == 12.0
    assert iterations[0]["system_gbest_distance"] == 12.0
    assert iterations[1]["system_gbest_distance"] == 12.0
    assert tracker.should_stop is False


def test_replay_preserves_early_stop_decision() -> None:
    tracker = SolutionProgressTracker(
        provider="groq",
        reference_distance=100.0,
        reference_type="tsplib_known_optimum",
        reference_is_proven_optimal=True,
        early_stop_policy=EarlyStopPolicy(
            threshold_percent=1.0,
        ),
    )
    initializer = {
        "source": "zero_shot",
        "iteration": 0,
        "route": [0, 1, 0],
        "validation": {"is_valid": True},
        "distance": 110.0,
        "gap_to_reference_percent": 10.0,
    }
    iterations = [
        {
            "iteration": 1,
            "route": [0, 1, 0],
            "validation": {"is_valid": True},
            "distance": 100.5,
            "gap_to_reference_percent": 0.5,
        }
    ]

    ma2._replay_solution_progress(
        tracker,
        initializer=initializer,
        iterations=iterations,
    )

    assert tracker.should_stop is True
    assert (
        tracker.latest_early_stop["reason"]
        == "gap_threshold_reached"
    )
