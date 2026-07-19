from pathlib import Path

from run_baseline import build_baseline_summary
from run_gemini_multi_agent2 import build_multi_agent2_summary
from run_gemini_zero_shot import build_zero_shot_summary


def test_baseline_summary_keeps_reference_solutions_and_timing() -> None:
    result = {
        "experiment": {
            "run_id": "run-1",
            "seed": 42,
            "num_locations_including_depot": 10,
        },
        "solutions": {
            "or_tools": {
                "route": [0, 1, 0],
                "distance": 2.0,
                "validation": {"is_valid": True},
            },
            "exact": {
                "route": [0, 1, 0],
                "distance": 2.0,
                "validation": {"is_valid": True},
            },
        },
        "metrics": {"or_tools_gap_to_exact_percent": 0.0},
        "timing": {"exact_brute_force_seconds": 1.5},
    }

    summary = build_baseline_summary(
        result,
        source_results=Path("baseline_results.json"),
    )

    assert summary["status"] == "completed"
    assert summary["seed"] == 42
    assert summary["or_tools"]["gap_to_exact_percent"] == 0.0
    assert summary["exact_brute_force"]["distance"] == 2.0
    assert summary["timing"]["exact_brute_force_seconds"] == 1.5


def test_zero_shot_summary_keeps_solution_api_and_error_status() -> None:
    result = {
        "experiment": "gemini_visual_zero_shot_tsp",
        "run_id": "run-1",
        "model": "gemini-test",
        "temperature": 0.0,
        "model_input": {"coordinates_sent_to_model": False},
        "route": [0, 1, 0],
        "validation": {
            "is_valid": True,
            "missing_nodes": [],
            "repeated_nodes": [],
        },
        "metrics": {
            "distance": 2.0,
            "gap_to_or_tools_percent": 0.0,
            "gap_to_exact_percent": 0.0,
        },
        "run_summary": {"api_call_count": 1, "total_token_count": 100},
        "timing": {"api_call_wall_seconds": 3.0},
        "errors": [
            {
                "phase": "zero_shot_initializer",
                "type": "ClientError",
                "message": "429 RESOURCE_EXHAUSTED",
            }
        ],
    }

    summary = build_zero_shot_summary(
        result,
        source_results=Path("gemini_zero_shot_results.json"),
    )

    assert summary["status"] == "completed"
    assert summary["coordinates_sent_to_model"] is False
    assert summary["solution"]["is_valid"] is True
    assert summary["solution"]["gap_to_exact_percent"] == 0.0
    assert summary["api_summary"]["api_call_count"] == 1
    assert summary["errors"][0]["status_code"] == 429


def test_multi_agent2_summary_counts_valid_and_optimal_iterations() -> None:
    result = {
        "experiment": "gemini_visual_multi_agent_2_tsp",
        "run_id": "run-1",
        "model": "gemini-test",
        "requested_iterations": 3,
        "completed_iterations": 2,
        "initializer": {
            "source": "zero-shot",
            "route": [0, 1, 0],
            "validation": {"is_valid": True},
            "distance": 2.0,
            "gap_to_exact_percent": 0.0,
        },
        "critic_iterations": [
            {
                "iteration": 1,
                "iteration_type": "critic_route_revision",
                "route": [0, 1, 0],
                "validation": {"is_valid": True},
                "distance": 2.0,
                "gap_to_exact_percent": 0.0,
                "api_call": {
                    "api_call_wall_seconds": 3.0,
                    "request_total_wall_seconds": 3.2,
                    "usage": {"total_token_count": 100},
                },
                "timing": {"iteration_total_wall_seconds": 3.5},
            },
            {
                "iteration": 2,
                "iteration_type": "critic_route_revision",
                "route": [0, 2, 0],
                "validation": {"is_valid": False},
                "distance": 3.0,
                "gap_to_exact_percent": 50.0,
                "api_call": {},
                "timing": {},
            },
        ],
        "final_solution": {
            "route": [0, 2, 0],
            "validation": {"is_valid": False},
            "distance": 3.0,
            "gap_to_exact_percent": 50.0,
        },
        "best_valid_solution": {
            "source": "critic",
            "iteration": 1,
            "route": [0, 1, 0],
            "validation": {"is_valid": True},
            "distance": 2.0,
            "gap_to_exact_percent": 0.0,
        },
        "run_summary": {"api_call_count": 2},
        "errors": [
            {
                "iteration": 3,
                "phase": "critic",
                "type": "ClientError",
                "message": "429 RESOURCE_EXHAUSTED",
            }
        ],
    }

    summary = build_multi_agent2_summary(
        result,
        source_results=Path("gemini_multi_agent2_results.json"),
    )

    assert summary["status"] == "partial"
    assert summary["quality_summary"]["valid_iterations"] == 1
    assert summary["quality_summary"]["valid_iteration_rate_percent"] == 50.0
    assert summary["quality_summary"]["optimal_iterations"] == 1
    assert summary["quality_summary"]["optimal_iteration_rate_percent"] == 50.0
    assert summary["iterations"][0]["total_token_count"] == 100
    assert summary["errors"][0]["status_code"] == 429
