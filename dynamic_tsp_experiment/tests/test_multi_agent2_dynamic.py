from pathlib import Path

import pytest

import run_gemini_multi_agent2 as ma2


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
