from pathlib import Path

import pytest

from run_gemini_multi_agent1 import build_multi_agent1_summary


def _candidate(candidate_id: int, *, valid: bool, gap: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "route": [0, 1, 0],
        "validation": {"is_valid": valid},
        "distance": 2.0,
        "gap_to_exact_percent": gap,
    }


def _api_call(seconds: float, tokens: int) -> dict:
    return {
        "api_call_wall_seconds": seconds,
        "request_total_wall_seconds": seconds + 0.2,
        "usage": {"total_token_count": tokens},
    }


def test_summary_compacts_quality_timing_and_error_fields() -> None:
    candidates = [
        _candidate(1, valid=True, gap=0.0),
        _candidate(2, valid=True, gap=12.5),
        _candidate(3, valid=False, gap=20.0),
    ]
    result = {
        "experiment": "gemini_visual_multi_agent_1_tsp",
        "run_id": "test-run",
        "model": "gemini-test",
        "requested_iterations": 1,
        "completed_iterations": 1,
        "iterations": [
            {
                "iteration": 1,
                "critic": {
                    "candidates": candidates,
                    "api_call": _api_call(2.0, 100),
                },
                "scorer": {
                    "best_candidate_id": 1,
                    "selected_is_oracle_best_after_evaluation": True,
                    "selection_regret_percent_after_evaluation": 0.0,
                    "api_calls": [_api_call(1.0, 50)],
                },
                "selected_solution": {
                    "route": [0, 1, 0],
                    "validation": {"is_valid": True},
                    "distance": 2.0,
                    "gap_to_exact_percent": 0.0,
                },
                "timing": {"logical_iteration_measured_seconds": 3.5},
            }
        ],
        "pending_iteration": None,
        "run_summary": {"api_call_count": 2, "total_token_count": 150},
        "errors": [
            {
                "iteration": 1,
                "phase": "scorer",
                "type": "ClientError",
                "message": "429 RESOURCE_EXHAUSTED",
                "api_call": {"api_call_wall_seconds": 0.5},
            }
        ],
    }

    summary = build_multi_agent1_summary(
        result,
        source_results=Path("results.json"),
    )

    assert summary["status"] == "completed"
    assert summary["source_results"] == "results.json"
    quality = summary["quality_summary"]
    assert quality["total_critic_candidates"] == 3
    assert quality["valid_critic_candidates"] == 2
    assert quality["valid_candidate_rate_percent"] == pytest.approx(200 / 3)
    assert quality["optimal_critic_candidates"] == 1
    assert quality["optimal_candidate_rate_percent"] == pytest.approx(100 / 3)
    assert quality["optimal_scorer_selection_rate_percent"] == 100.0
    assert summary["iterations"][0]["critic"]["nonoptimal_candidate_ids"] == [2]
    assert summary["iterations"][0]["scorer"]["total_token_count"] == 50
    assert summary["errors"][0]["status_code"] == 429


def test_summary_marks_pending_experiment_as_partial() -> None:
    result = {
        "experiment": "gemini_visual_multi_agent_1_tsp",
        "run_id": "partial-run",
        "model": "gemini-test",
        "requested_iterations": 2,
        "completed_iterations": 0,
        "iterations": [],
        "pending_iteration": {
            "iteration": 1,
            "critic": {
                "candidates": [_candidate(1, valid=True, gap=0.0)]
            },
        },
        "run_summary": {},
        "errors": [],
    }

    summary = build_multi_agent1_summary(
        result,
        source_results=Path("partial.json"),
    )

    assert summary["status"] == "partial"
    assert summary["pending_iteration"] == {
        "iteration": 1,
        "phase": "scorer",
        "critic_candidate_count": 1,
    }
    assert summary["quality_summary"]["total_critic_candidates"] == 1
    assert summary["quality_summary"]["optimal_critic_candidates"] == 1
