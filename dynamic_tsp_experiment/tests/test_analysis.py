import json
from pathlib import Path

import pytest

from src.analysis import build_analysis


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def _evaluation(
    distance: float,
    *,
    valid: bool = True,
) -> dict:
    return {
        "validation": {"is_valid": valid},
        "distance": distance,
        "reference_distance": 100.0,
        "gap_to_reference_percent": (
            distance - 100.0
            if valid
            else None
        ),
    }


def _manifest() -> dict:
    return {
        "schema_version": "1.0",
        "run_id": "test_run",
        "problem": {
            "name": "random_n5_seed42",
            "source_type": "random",
            "dimension": 5,
            "depot_id": 0,
            "edge_weight_type": "EUC_2D_CONTINUOUS",
            "fingerprint_sha256": "fingerprint",
            "reference": {
                "type": "or_tools_heuristic",
                "distance": 100.0,
                "is_proven_optimal": False,
            },
        },
    }


def _identity(method: str) -> dict:
    return {
        "schema_version": "2.0",
        "run_id": "test_run",
        "method": method,
        "problem": {
            "fingerprint_sha256": "fingerprint",
        },
    }


def test_analysis_contains_every_iteration_but_not_raw_payloads(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "test_run"
    baseline = {
        **_identity("baseline"),
        "or_tools": _evaluation(100.0),
        "timing": {"or_tools_wall_seconds": 1.0},
    }
    zero = {
        **_identity("zero_shot"),
        **_evaluation(120.0),
        "model": {"name": "gemini"},
        "raw_response": "çok uzun cevap",
        "timing": {"api_call_wall_seconds": 2.0},
        "run_summary": {
            "api_call_count": 1,
            "total_token_count": 50,
        },
        "errors": [],
    }
    ma2 = {
        **_identity("multi_agent_2"),
        "requested_iterations": 2,
        "completed_iterations": 2,
        "initializer": {
            "source": "zero_shot",
            "iteration": 0,
            **_evaluation(120.0),
        },
        "final_solution": {
            "source": "critic",
            "iteration": 2,
            **_evaluation(110.0),
        },
        "best_valid_solution": {
            "source": "critic",
            "iteration": 1,
            **_evaluation(105.0),
        },
        "iterations": [
            {
                "iteration": 1,
                **_evaluation(105.0),
                "api_call": {
                    "usage": {"total_token_count": 10}
                },
                "timing": {
                    "api_call_wall_seconds": 1.0,
                    "iteration_total_wall_seconds": 1.2,
                },
            },
            {
                "iteration": 2,
                **_evaluation(110.0),
                "api_call": {
                    "usage": {"total_token_count": 11}
                },
                "timing": {
                    "api_call_wall_seconds": 1.1,
                    "iteration_total_wall_seconds": 1.3,
                },
            },
        ],
        "errors": [],
    }
    selected = {
        **_evaluation(103.0),
        "artifacts": {"route_image": "image.png"},
    }
    candidates = [
        {
            "candidate_id": 1,
            **_evaluation(103.0),
            "raw_response": "uzun",
        },
        {
            "candidate_id": 2,
            **_evaluation(90.0, valid=False),
            "raw_response": "uzun",
        },
    ]
    ma1 = {
        **_identity("multi_agent_1"),
        "candidate_count_requested": 2,
        "requested_iterations": 1,
        "completed_iterations": 1,
        "initializer": {
            "source": "zero_shot",
            "iteration": 0,
            **_evaluation(120.0),
        },
        "final_solution": {
            "source": "scorer_selection",
            "iteration": 1,
            **selected,
        },
        "best_valid_solution": {
            "source": "scorer_selection",
            "iteration": 1,
            **selected,
        },
        "best_critic_candidate_oracle": {
            "source": "critic_candidate_oracle",
            "iteration": 1,
            "candidate_id": 1,
            **_evaluation(103.0),
        },
        "pending_iteration": None,
        "iterations": [
            {
                "iteration": 1,
                "critic": {
                    "returned_candidate_count": 2,
                    "candidates": candidates,
                    "api_call": {
                        "usage": {"total_token_count": 20}
                    },
                    "timing": {
                        "api_call_wall_seconds": 2.0
                    },
                },
                "scorer": {
                    "selection_mode": (
                        "visual_scorer_after_feasibility_filter"
                    ),
                    "best_candidate_id": 1,
                    "selection_regret_percent_after_evaluation": 0.0,
                    "api_call": {
                        "usage": {"total_token_count": 5}
                    },
                    "timing": {
                        "api_call_wall_seconds": 0.5
                    },
                },
                "selected_solution": selected,
                "timing": {
                    "critic_stage_wall_seconds": 2.2,
                    "scorer_stage_wall_seconds": 0.7,
                    "iteration_processing_wall_seconds": 3.0,
                },
            }
        ],
        "errors": [],
    }
    _write(
        run_dir / "baseline" / "baseline_results.json",
        baseline,
    )
    _write(
        run_dir / "zero_shot" / "zero_shot_results.json",
        zero,
    )
    _write(
        run_dir / "multi_agent1" / "multi_agent1_results.json",
        ma1,
    )
    _write(
        run_dir / "multi_agent2" / "multi_agent2_results.json",
        ma2,
    )

    analysis = build_analysis(
        run_dir=run_dir,
        manifest=_manifest(),
    )

    assert analysis["completion"]["all_methods_completed"]
    assert len(
        analysis["methods"]["multi_agent_2"]["iterations"]
    ) == 2
    assert len(
        analysis["methods"]["multi_agent_1"]["iterations"]
    ) == 1
    assert (
        analysis["comparison"]["best_valid_mllm_solution"][
            "method"
        ]
        == "multi_agent_1_best_valid"
    )
    ma1_analysis = analysis["methods"]["multi_agent_1"]
    assert ma1_analysis["valid_candidate_rate_percent"] == 50.0
    assert ma1_analysis[
        "scorer_best_candidate_selection_rate_percent"
    ] == 100.0
    assert ma1_analysis["iterations"][0][
        "selected_best_valid_candidate"
    ] is True
    encoded = json.dumps(analysis)
    assert "raw_response" not in encoded
    assert "coordinates" not in encoded
    assert "route_image" not in encoded


def test_analysis_supports_partial_runs(
    tmp_path: Path,
) -> None:
    analysis = build_analysis(
        run_dir=tmp_path / "runs" / "test_run",
        manifest=_manifest(),
    )
    assert not analysis["completion"]["all_methods_completed"]
    assert (
        analysis["methods"]["multi_agent_1"]["status"]
        == "not_run"
    )


def test_analysis_rejects_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "test_run"
    result = {
        **_identity("zero_shot"),
        "problem": {"fingerprint_sha256": "wrong"},
    }
    _write(
        run_dir / "zero_shot" / "zero_shot_results.json",
        result,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        build_analysis(
            run_dir=run_dir,
            manifest=_manifest(),
        )


def test_analysis_discovers_unified_provider_layout(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "test_run"
    baseline = {
        **_identity("baseline"),
        "or_tools": _evaluation(100.0),
        "timing": {},
    }
    model = {
        "provider": "groq",
        "alias": "qwen/qwen3.6-27b",
        "requested_name": "qwen/qwen3.6-27b",
    }
    zero = {
        **_identity("zero_shot"),
        "model": model,
        **_evaluation(120.0),
        "timing": {"api_call_wall_seconds": 2.0},
        "run_summary": {
            "api_call_count": 1,
            "total_token_count": 10,
        },
        "errors": [],
    }
    ma2 = {
        **_identity("multi_agent_2"),
        "model": model,
        "requested_iterations": 1,
        "completed_iterations": 1,
        "initializer": _evaluation(120.0),
        "final_solution": _evaluation(110.0),
        "best_valid_solution": _evaluation(110.0),
        "iterations": [
            {
                "iteration": 1,
                **_evaluation(110.0),
                "timing": {
                    "api_call_wall_seconds": 1.0,
                    "iteration_total_wall_seconds": 1.2,
                },
            }
        ],
        "errors": [],
    }
    ma1 = {
        **_identity("multi_agent_1"),
        "model": model,
        "candidate_count_requested": 2,
        "requested_iterations": 1,
        "completed_iterations": 1,
        "pending_iteration": None,
        "initializer": _evaluation(120.0),
        "final_solution": _evaluation(105.0),
        "best_valid_solution": _evaluation(105.0),
        "best_critic_candidate_oracle": _evaluation(103.0),
        "iterations": [
            {
                "iteration": 1,
                "critic": {
                    "returned_candidate_count": 2,
                    "candidates": [
                        {
                            "candidate_id": 1,
                            **_evaluation(103.0),
                        },
                        {
                            "candidate_id": 2,
                            **_evaluation(105.0),
                        },
                    ],
                },
                "scorer": {
                    "selection_mode": (
                        "visual_scorer_after_feasibility_filter"
                    ),
                    "best_candidate_id": 2,
                },
                "selected_solution": _evaluation(105.0),
                "timing": {},
            }
        ],
        "errors": [],
    }
    _write(
        run_dir / "baseline" / "baseline_results.json",
        baseline,
    )
    model_root = (
        run_dir
        / "providers"
        / "groq"
        / "qwen-qwen3.6-27b"
    )
    _write(
        model_root / "zero_shot" / "zero_shot_results.json",
        zero,
    )
    _write(
        model_root / "multi_agent1" / "multi_agent1_results.json",
        ma1,
    )
    _write(
        model_root / "multi_agent2" / "multi_agent2_results.json",
        ma2,
    )

    analysis = build_analysis(
        run_dir=run_dir,
        manifest=_manifest(),
    )

    assert analysis["completion"]["all_methods_completed"]
    assert analysis["completion"][
        "all_provider_models_completed"
    ]
    assert len(analysis["provider_models"]) == 1
    provider = analysis["provider_models"][0]
    assert provider["provider"] == "groq"
    assert provider["model_alias"] == "qwen/qwen3.6-27b"
    assert provider["layout"] == "unified_provider"
    assert provider["methods"]["multi_agent_1"][
        "scorer_best_candidate_selection_rate_percent"
    ] == 0.0
    rows = analysis["comparison"]["all_model_method_rows"]
    assert {row["method"] for row in rows} == {
        "zero_shot",
        "multi_agent_1",
        "multi_agent_2",
    }
