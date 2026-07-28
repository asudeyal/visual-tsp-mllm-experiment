from pathlib import Path

import pytest

import run_openrouter_multi_agent1 as ma1
import run_openrouter_multi_agent2 as ma2
from src.core import write_json
from src.problem_loader import generate_random_problem


ALIAS = "nemotron-3-nano-omni"


def _initializer_fixture(
    run_dir: Path,
    *,
    fingerprint: str,
) -> None:
    model_root = ma2._model_root(run_dir, ALIAS)
    image = model_root / "images" / "route.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    write_json(
        model_root / "zero_shot_results.json",
        {
            "problem": {
                "fingerprint_sha256": fingerprint,
            },
            "route": [0, 1, 2, 3, 0],
            "artifacts": {
                "route_image": image.relative_to(
                    run_dir
                ).as_posix(),
            },
        },
    )


def test_openrouter_methods_use_model_specific_directories(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run1"

    assert ma1._model_root(run_dir, ALIAS) == (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / ALIAS
    )
    assert ma2._model_root(run_dir, ALIAS) == (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / ALIAS
    )


def test_both_methods_load_same_model_zero_shot_initializer(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run1"
    problem = generate_random_problem(4, seed=42)
    fingerprint = "same-problem"
    _initializer_fixture(
        run_dir,
        fingerprint=fingerprint,
    )

    initializer2, route2, image2 = ma2._load_initializer(
        run_dir=run_dir,
        model_alias=ALIAS,
        problem=problem,
        fingerprint=fingerprint,
    )
    initializer1, route1, image1 = ma1._load_initializer(
        run_dir=run_dir,
        model_alias=ALIAS,
        problem=problem,
        fingerprint=fingerprint,
    )

    assert route1 == route2 == [0, 1, 2, 3, 0]
    assert image1 == image2
    assert initializer1["validation"]["is_valid"]
    assert initializer2["validation"]["is_valid"]


def test_invalid_initializer_is_preserved_for_critic_repair(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "run1"
    problem = generate_random_problem(4, seed=42)
    fingerprint = "same-problem"
    _initializer_fixture(
        run_dir,
        fingerprint=fingerprint,
    )
    result_path = (
        ma2._model_root(run_dir, ALIAS)
        / "zero_shot_results.json"
    )
    value = {
        "problem": {
            "fingerprint_sha256": fingerprint,
        },
        "route": [0, 1, 2, 2, 3, 0],
        "artifacts": {
            "route_image": (
                ma2._model_root(run_dir, ALIAS)
                / "images"
                / "route.png"
            )
            .relative_to(run_dir)
            .as_posix(),
        },
    }
    write_json(result_path, value)

    initializer, route, _ = ma2._load_initializer(
        run_dir=run_dir,
        model_alias=ALIAS,
        problem=problem,
        fingerprint=fingerprint,
    )

    assert route == [0, 1, 2, 2, 3, 0]
    assert initializer["validation"]["is_valid"] is False


def test_ma1_checkpoint_rejects_different_candidate_strategy() -> None:
    checkpoint = {
        "run_id": "run1",
        "model": "provider/model",
        "candidate_count_requested": 7,
        "candidate_strategy": "native_multiple_choices",
        "problem_fingerprint_sha256": "same",
    }

    with pytest.raises(ValueError, match="strateji"):
        ma1._validate_checkpoint(
            checkpoint,
            run_id="run1",
            model="provider/model",
            candidate_count=7,
            candidate_strategy="independent_calls",
            fingerprint="same",
        )


def test_ma1_summary_counts_independent_http_calls() -> None:
    critic_calls = [
        {
            "phase": f"critic_{index}",
            "started_at_utc": f"start-{index}",
            "finished_at_utc": f"finish-{index}",
        }
        for index in (1, 2)
    ]
    scorer_call = {
        "phase": "scorer",
        "started_at_utc": "start-3",
        "finished_at_utc": "finish-3",
    }
    iterations = [
        {
            "critic": {
                "api_call": {"phase": "aggregate"},
                "api_calls": critic_calls,
            },
            "scorer": {
                "api_call": scorer_call,
            },
        }
    ]

    calls = ma1._calls(iterations, None, [])

    assert calls == [*critic_calls, scorer_call]
