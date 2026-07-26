"""Uzun deney JSON'larından okunabilir küçük özetler üretir."""

from __future__ import annotations

from typing import Any


def _solution(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    result = {
        key: value.get(key)
        for key in (
            "source",
            "iteration",
            "candidate_id",
            "route",
            "distance",
            "gap_to_known_optimum_percent",
        )
        if key in value
    }
    validation = value.get("validation") or {}
    result["is_valid"] = validation.get("is_valid")
    result["is_optimal"] = (
        validation.get("is_valid") is True
        and value.get("gap_to_known_optimum_percent") is not None
        and abs(float(value["gap_to_known_optimum_percent"])) < 1e-9
    )
    return result


def baseline_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": result.get("experiment"),
        "run_id": result.get("run_id"),
        "instance": result.get("instance"),
        "known_optimum": _solution(result.get("known_optimum")),
        "or_tools": _solution(result.get("or_tools")),
        "timing": result.get("timing"),
    }


def zero_shot_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": result.get("experiment"),
        "run_id": result.get("run_id"),
        "method": "zero_shot",
        "model": result.get("model"),
        "temperature": result.get("temperature"),
        "solution": _solution(result),
        "run_summary": result.get("run_summary"),
        "timing": result.get("timing"),
        "errors": result.get("errors", []),
    }


def multi_agent2_summary(result: dict[str, Any]) -> dict[str, Any]:
    iterations = result.get("critic_iterations", [])
    valid = [item for item in iterations if item.get("validation", {}).get("is_valid")]
    optimal = [
        item
        for item in valid
        if item.get("gap_to_known_optimum_percent") is not None
        and abs(float(item["gap_to_known_optimum_percent"])) < 1e-9
    ]
    return {
        "experiment": result.get("experiment"),
        "run_id": result.get("run_id"),
        "method": "multi_agent_2",
        "model": result.get("model"),
        "requested_iterations": result.get("requested_iterations"),
        "completed_iterations": result.get("completed_iterations"),
        "valid_iteration_count": len(valid),
        "optimal_iteration_count": len(optimal),
        "initializer": _solution(result.get("initializer")),
        "final_solution": _solution(result.get("final_solution")),
        "best_valid_solution": _solution(result.get("best_valid_solution")),
        "iterations": [
            {
                "iteration": item.get("iteration"),
                "distance": item.get("distance"),
                "gap_percent": item.get("gap_to_known_optimum_percent"),
                "is_valid": item.get("validation", {}).get("is_valid"),
                "api_call_wall_seconds": item.get("api_call", {}).get(
                    "api_call_wall_seconds"
                ),
            }
            for item in iterations
        ],
        "run_summary": result.get("run_summary"),
        "errors": result.get("errors", []),
    }


def multi_agent1_summary(result: dict[str, Any]) -> dict[str, Any]:
    iterations = result.get("iterations", [])
    candidates = [
        candidate
        for iteration in iterations
        for candidate in iteration.get("critic", {}).get("candidates", [])
    ]
    valid_candidates = [
        item for item in candidates if item.get("validation", {}).get("is_valid")
    ]
    optimal_candidates = [
        item
        for item in valid_candidates
        if item.get("gap_to_known_optimum_percent") is not None
        and abs(float(item["gap_to_known_optimum_percent"])) < 1e-9
    ]
    optimal_selections = sum(
        1
        for iteration in iterations
        if iteration.get("selected_solution", {})
        .get("validation", {})
        .get("is_valid")
        is True
        and iteration.get("selected_solution", {}).get(
            "gap_to_known_optimum_percent"
        )
        is not None
        and abs(
            float(
                iteration["selected_solution"]["gap_to_known_optimum_percent"]
            )
        )
        < 1e-9
    )
    return {
        "experiment": result.get("experiment"),
        "run_id": result.get("run_id"),
        "method": "multi_agent_1",
        "scorer_policy": result.get("scorer_policy"),
        "model": result.get("model"),
        "candidate_count_requested": result.get("candidate_count_requested"),
        "requested_iterations": result.get("requested_iterations"),
        "completed_iterations": result.get("completed_iterations"),
        "candidate_count_total": len(candidates),
        "valid_candidate_count": len(valid_candidates),
        "optimal_candidate_count": len(optimal_candidates),
        "optimal_scorer_selection_count": optimal_selections,
        "initializer": _solution(result.get("initializer")),
        "final_solution": _solution(result.get("final_solution")),
        "best_valid_solution": _solution(result.get("best_valid_solution")),
        "best_critic_candidate_oracle": _solution(
            result.get("best_critic_candidate_oracle")
        ),
        "iterations": [
            {
                "iteration": item.get("iteration"),
                "returned_candidate_count": item.get("critic", {}).get(
                    "returned_candidate_count"
                ),
                "selected_candidate_id": item.get("scorer", {}).get(
                    "best_candidate_id"
                ),
                "selection_mode": item.get("scorer", {}).get("selection_mode"),
                "eligible_candidate_ids": item.get("scorer", {}).get(
                    "eligible_candidate_ids"
                ),
                "excluded_invalid_candidate_ids": item.get("scorer", {}).get(
                    "excluded_invalid_candidate_ids"
                ),
                "selected_distance": item.get("selected_solution", {}).get(
                    "distance"
                ),
                "selected_gap_percent": item.get("selected_solution", {}).get(
                    "gap_to_known_optimum_percent"
                ),
                "selection_regret_percent": item.get("scorer", {}).get(
                    "selection_regret_percent_after_evaluation"
                ),
                "critic_api_seconds": (
                    item.get("critic", {}).get("api_call") or {}
                ).get("api_call_wall_seconds"),
                "scorer_api_seconds": (
                    item.get("scorer", {}).get("api_call") or {}
                ).get("api_call_wall_seconds"),
            }
            for item in iterations
        ],
        "pending_iteration": (
            result.get("pending_iteration", {}).get("iteration")
            if result.get("pending_iteration")
            else None
        ),
        "run_summary": result.get("run_summary"),
        "errors": result.get("errors", []),
    }
