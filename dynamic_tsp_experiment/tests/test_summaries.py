from src.summaries import multi_agent1_summary, multi_agent2_summary


def _evaluation(distance: int = 426) -> dict:
    return {
        "route": [1, 2, 1],
        "validation": {"is_valid": True},
        "legal_node_ids": True,
        "distance": distance,
        "gap_to_known_optimum_percent": 100 * (distance - 426) / 426,
    }


def test_multi_agent2_summary_counts_valid_optimal_iterations() -> None:
    item = {"iteration": 1, "api_call": {"api_call_wall_seconds": 2.0}, **_evaluation()}
    summary = multi_agent2_summary(
        {
            "experiment": "x",
            "critic_iterations": [item],
            "initializer": _evaluation(),
            "final_solution": _evaluation(),
            "best_valid_solution": _evaluation(),
        }
    )
    assert summary["valid_iteration_count"] == 1
    assert summary["optimal_iteration_count"] == 1


def test_multi_agent1_summary_counts_candidates_and_selection() -> None:
    candidate = {"candidate_id": 1, **_evaluation()}
    result = {
        "experiment": "x",
        "iterations": [
            {
                "iteration": 1,
                "critic": {"returned_candidate_count": 1, "candidates": [candidate]},
                "scorer": {"best_candidate_id": 1},
                "selected_solution": _evaluation(),
            }
        ],
        "initializer": _evaluation(),
        "final_solution": _evaluation(),
        "best_valid_solution": _evaluation(),
        "best_critic_candidate_oracle": candidate,
    }
    summary = multi_agent1_summary(result)
    assert summary["candidate_count_total"] == 1
    assert summary["optimal_candidate_count"] == 1
    assert summary["optimal_scorer_selection_count"] == 1
