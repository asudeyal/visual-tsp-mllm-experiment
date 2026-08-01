from __future__ import annotations

import pytest

from src.solution_tracking import (
    EarlyStopPolicy,
    SolutionObservation,
    SolutionProgressTracker,
    best_valid_observation,
    selected_is_iteration_best,
    selection_regret_percent,
)


def solution(
    distance: float,
    *,
    gap: float | None = None,
    valid: bool = True,
    route: list[int] | None = None,
    candidate_index: int | None = None,
) -> dict:
    result = {
        "route": route or [0, 1, 2, 0],
        "distance": distance,
        "validation": {
            "is_valid": valid,
        },
    }

    if gap is not None:
        result["gap_to_reference_percent"] = gap

    if candidate_index is not None:
        result["candidate_index"] = candidate_index

    return result


def make_tracker(
    *,
    provider: str = "gemini",
    reference_distance: float = 100.0,
    proven: bool = True,
    threshold: float = 1.0,
) -> SolutionProgressTracker:
    return SolutionProgressTracker(
        provider=provider,
        reference_distance=reference_distance,
        reference_type=(
            "tsplib_known_optimum"
            if proven
            else "or_tools_heuristic"
        ),
        reference_is_proven_optimal=proven,
        early_stop_policy=EarlyStopPolicy(
            threshold_percent=threshold,
        ),
    )


def test_best_valid_observation_ignores_invalid_routes() -> None:
    observations = [
        SolutionObservation.from_mapping(
            solution(80.0, valid=False),
            source="critic",
            iteration=1,
            candidate_index=1,
        ),
        SolutionObservation.from_mapping(
            solution(110.0),
            source="critic",
            iteration=1,
            candidate_index=2,
        ),
        SolutionObservation.from_mapping(
            solution(105.0),
            source="critic",
            iteration=1,
            candidate_index=3,
        ),
    ]

    best = best_valid_observation(observations)

    assert best is not None
    assert best.distance == pytest.approx(105.0)
    assert best.candidate_index == 3


def test_selection_regret_and_correct_selection() -> None:
    selected = SolutionObservation.from_mapping(
        solution(95.0),
        source="selected",
        iteration=1,
        candidate_index=1,
    )
    best = SolutionObservation.from_mapping(
        solution(90.0),
        source="critic",
        iteration=1,
        candidate_index=2,
    )

    assert selection_regret_percent(
        selected,
        best,
    ) == pytest.approx((5.0 / 90.0) * 100.0)

    assert selected_is_iteration_best(selected, best) is False


def test_equal_distance_counts_as_correct_selection() -> None:
    selected = SolutionObservation.from_mapping(
        solution(90.0),
        source="selected",
        iteration=1,
        candidate_index=2,
    )
    best = SolutionObservation.from_mapping(
        solution(90.0),
        source="critic",
        iteration=1,
        candidate_index=1,
    )

    assert selected_is_iteration_best(selected, best) is True
    assert selection_regret_percent(
        selected,
        best,
    ) == pytest.approx(0.0)


def test_multi_agent1_keeps_system_and_oracle_gbest_separate() -> None:
    tracker = make_tracker(
        reference_distance=80.0,
    )

    tracker.seed_initializer(
        solution(100.0, gap=25.0),
    )

    progress = tracker.record_iteration(
        iteration=1,
        selected_solution=solution(
            95.0,
            gap=18.75,
            candidate_index=1,
        ),
        candidates=[
            solution(95.0, gap=18.75),
            solution(90.0, gap=12.5),
        ],
    )

    assert progress["iteration_best"]["distance"] == pytest.approx(
        90.0
    )
    assert progress["system_gbest"]["distance"] == pytest.approx(
        95.0
    )
    assert progress[
        "observed_candidate_gbest"
    ]["distance"] == pytest.approx(90.0)

    assert progress["selected_is_iteration_best"] is False
    assert progress["selection_regret_percent"] == pytest.approx(
        (5.0 / 90.0) * 100.0
    )


def test_invalid_selected_solution_does_not_replace_system_gbest() -> None:
    tracker = make_tracker()

    tracker.seed_initializer(
        solution(120.0, gap=20.0),
    )

    progress = tracker.record_iteration(
        iteration=1,
        selected_solution=solution(
            80.0,
            valid=False,
        ),
        candidates=[
            solution(80.0, valid=False),
            solution(110.0, gap=10.0),
        ],
    )

    assert progress["system_gbest"]["distance"] == pytest.approx(
        120.0
    )
    assert progress["iteration_best"]["distance"] == pytest.approx(
        110.0
    )
    assert progress["selected_is_iteration_best"] is False
    assert progress["selection_regret_percent"] is None


def test_proven_optimum_gap_triggers_early_stop() -> None:
    tracker = make_tracker(
        provider="gemini",
        reference_distance=100.0,
        proven=True,
    )

    tracker.seed_initializer(
        solution(110.0, gap=10.0),
    )

    progress = tracker.record_iteration(
        iteration=1,
        selected_solution=solution(
            100.8,
            gap=0.8,
        ),
        candidates=[
            solution(100.8, gap=0.8),
        ],
    )

    assert progress["early_stop"]["eligible"] is True
    assert progress["early_stop"]["should_stop"] is True
    assert (
        progress["early_stop"]["reason"]
        == "gap_threshold_reached"
    )


def test_initializer_can_trigger_early_stop_before_iterations() -> None:
    tracker = make_tracker(
        provider="groq",
        reference_distance=100.0,
        proven=True,
    )

    state = tracker.seed_initializer(
        solution(100.5, gap=0.5),
    )

    assert state["early_stop"]["should_stop"] is True
    assert tracker.should_stop is True
    assert (
        state["early_stop"]["system_gbest_iteration"]
        == 0
    )


def test_heuristic_reference_does_not_trigger_early_stop() -> None:
    tracker = make_tracker(
        provider="gemini",
        reference_distance=100.0,
        proven=False,
    )

    state = tracker.seed_initializer(
        solution(100.2, gap=0.2),
    )

    assert state["early_stop"]["eligible"] is False
    assert state["early_stop"]["should_stop"] is False
    assert (
        state["early_stop"]["reason"]
        == "reference_not_proven_optimal"
    )


def test_openrouter_is_not_enabled_for_early_stop() -> None:
    tracker = make_tracker(
        provider="openrouter",
        reference_distance=100.0,
        proven=True,
    )

    state = tracker.seed_initializer(
        solution(100.2, gap=0.2),
    )

    assert state["early_stop"]["eligible"] is False
    assert state["early_stop"]["should_stop"] is False
    assert (
        state["early_stop"]["reason"]
        == "provider_not_enabled"
    )


def test_negative_gap_does_not_trigger_false_optimum_claim() -> None:
    tracker = make_tracker(
        provider="gemini",
        reference_distance=100.0,
        proven=True,
    )

    state = tracker.seed_initializer(
        solution(90.0, gap=-10.0),
    )

    assert state["early_stop"]["should_stop"] is False
    assert (
        state["early_stop"]["reason"]
        == "negative_gap_inconsistent"
    )


def test_resume_replay_reconstructs_same_gbest() -> None:
    tracker = make_tracker(
        reference_distance=100.0,
    )

    tracker.seed_initializer(
        solution(130.0, gap=30.0),
    )

    tracker.record_iteration(
        iteration=1,
        selected_solution=solution(120.0, gap=20.0),
        candidates=[
            solution(120.0, gap=20.0),
            solution(115.0, gap=15.0),
        ],
    )

    tracker.record_iteration(
        iteration=2,
        selected_solution=solution(110.0, gap=10.0),
        candidates=[
            solution(110.0, gap=10.0),
            solution(105.0, gap=5.0),
        ],
    )

    snapshot = tracker.snapshot()

    replayed = make_tracker(
        reference_distance=100.0,
    )

    replayed.seed_initializer(
        solution(130.0, gap=30.0),
    )

    replayed.record_iteration(
        iteration=1,
        selected_solution=solution(120.0, gap=20.0),
        candidates=[
            solution(120.0, gap=20.0),
            solution(115.0, gap=15.0),
        ],
    )

    replayed.record_iteration(
        iteration=2,
        selected_solution=solution(110.0, gap=10.0),
        candidates=[
            solution(110.0, gap=10.0),
            solution(105.0, gap=5.0),
        ],
    )

    replayed_snapshot = replayed.snapshot()

    assert replayed_snapshot["system_gbest"] == snapshot[
        "system_gbest"
    ]
    assert replayed_snapshot[
        "observed_candidate_gbest"
    ] == snapshot["observed_candidate_gbest"]
    assert replayed_snapshot["history"] == snapshot["history"]


def test_multi_agent2_tracks_best_accepted_solution() -> None:
    tracker = make_tracker(
        reference_distance=100.0,
    )

    tracker.seed_initializer(
        solution(130.0, gap=30.0),
    )

    first = tracker.record_multi_agent2_iteration(
        iteration=1,
        solution=solution(120.0, gap=20.0),
    )

    second = tracker.record_multi_agent2_iteration(
        iteration=2,
        solution=solution(125.0, gap=25.0),
    )

    assert first["system_gbest"]["distance"] == pytest.approx(
        120.0
    )
    assert second["system_gbest"]["distance"] == pytest.approx(
        120.0
    )