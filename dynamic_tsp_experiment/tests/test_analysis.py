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


def _observability(*, retries: int = 1) -> dict:
    metrics = {
        "system_cpu_percent": {
            "sample_count": 2,
            "average": 25.0,
            "maximum": 40.0,
        },
        "process_memory_rss_mb": {
            "sample_count": 2,
            "average": 100.0,
            "maximum": 120.0,
        },
    }
    return {
        "settings": {
            "profile_resources": True,
            "resource_sample_interval_seconds": 0.5,
            "minimum_request_interval_seconds": 1.0,
            "max_retries": 2,
            "early_stop_enabled": True,
            "early_stop_gap_percent": 1.0,
        },
        "request_control": {
            "execution_count": 2,
            "request_attempt_count": 2 + retries,
            "retry_count": retries,
            "active_wall_seconds": 2.5,
            "total_wall_seconds": 4.0,
            "waits": {
                "deliberate_delay_seconds": 1.0,
                "rate_limit_backoff_seconds": 0.5,
                "controlled_wait_seconds": 1.5,
            },
        },
        "resources": {
            "enabled": True,
            "duration_seconds": 4.0,
            "sample_count": 2,
            "system": {
                "local_gpu": {
                    "available": False,
                    "backend": "nvidia_nvml",
                    "unavailable_reason": "NVML unavailable",
                }
            },
            "overall": {"metrics": metrics},
            "by_phase": {
                "api_request": {
                    "sample_count": 2,
                    "metrics": metrics,
                }
            },
            "sampling_errors": [],
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
        "observability": _observability(retries=0),
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
        "observability": _observability(),
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
                "token_count": 10,
                "iteration_best_distance": 105.0,
                "system_gbest_distance": 105.0,
                "system_gbest_gap_percent": 5.0,
                "api_call": {
                    "usage": {"total_token_count": 10}
                },
                "timing": {
                    "api_call_wall_seconds": 1.0,
                    "api_active_wall_seconds": 0.9,
                    "deliberate_delay_seconds": 0.2,
                    "rate_limit_backoff_seconds": 0.1,
                    "controlled_wait_seconds": 0.3,
                    "iteration_active_wall_seconds": 1.1,
                    "iteration_total_wall_seconds": 1.2,
                },
            },
            {
                "iteration": 2,
                **_evaluation(110.0),
                "token_count": 11,
                "iteration_best_distance": 110.0,
                "system_gbest_distance": 105.0,
                "system_gbest_gap_percent": 5.0,
                "api_call": {
                    "usage": {"total_token_count": 11}
                },
                "timing": {
                    "api_call_wall_seconds": 1.1,
                    "api_active_wall_seconds": 1.0,
                    "deliberate_delay_seconds": 0.0,
                    "rate_limit_backoff_seconds": 0.0,
                    "controlled_wait_seconds": 0.0,
                    "iteration_active_wall_seconds": 1.3,
                    "iteration_total_wall_seconds": 1.3,
                },
            },
        ],
        "termination": {
            "reason": "requested_iterations_completed",
            "early_stop": None,
            "failed_iteration": None,
        },
        "observability": _observability(),
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
                "token_count": 25,
                "iteration_best_distance": 103.0,
                "system_gbest_distance": 103.0,
                "system_gbest_gap_percent": 3.0,
                "observed_candidate_gbest_distance": 103.0,
                "selection_regret_percent": 0.0,
                "selected_is_iteration_best": True,
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
                    "local_processing_active_wall_seconds": 0.5,
                    "iteration_active_wall_seconds": 2.8,
                    "deliberate_delay_seconds": 0.1,
                    "rate_limit_backoff_seconds": 0.1,
                    "controlled_wait_seconds": 0.2,
                    "iteration_observed_total_wall_seconds": 3.0,
                },
            }
        ],
        "termination": {
            "reason": "requested_iterations_completed",
            "early_stop": None,
            "failed_iteration": None,
        },
        "observability": _observability(),
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
    assert ma1_analysis["iterations"][0][
        "system_gbest_distance"
    ] == 103.0
    assert ma1_analysis["iterations"][0][
        "observed_candidate_gbest_distance"
    ] == 103.0
    assert ma1_analysis["iterations"][0]["token_count"][
        "total"
    ] == 25
    assert ma1_analysis["timing_seconds"][
        "rate_limit_backoff"
    ] == pytest.approx(0.1)
    ma2_analysis = analysis["methods"]["multi_agent_2"]
    assert ma2_analysis["iterations"][1][
        "system_gbest_distance"
    ] == 105.0
    assert ma2_analysis["total_token_count"] == 21
    runtime = ma1_analysis["observability"]
    assert runtime["request_control"]["retry_count"] == 1
    assert runtime["request_control"][
        "deliberate_delay_seconds"
    ] == pytest.approx(1.0)
    assert runtime["resources"]["overall_metrics"][
        "system_cpu_percent"
    ]["average"] == pytest.approx(25.0)
    assert runtime["resources"]["phases"]["api_request"][
        "sample_count"
    ] == 2
    encoded = json.dumps(analysis)
    assert "raw_response" not in encoded
    assert "coordinates" not in encoded
    assert "route_image" not in encoded
    assert '"events"' not in encoded
    assert '"executions"' not in encoded


def test_early_stop_is_a_successful_iterative_completion(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "test_run"
    result = {
        **_identity("multi_agent_2"),
        "requested_iterations": 10,
        "completed_iterations": 2,
        "initializer": _evaluation(120.0),
        "final_solution": _evaluation(100.5),
        "best_valid_solution": _evaluation(100.5),
        "iterations": [
            {"iteration": 1, **_evaluation(105.0), "timing": {}},
            {"iteration": 2, **_evaluation(100.5), "timing": {}},
        ],
        "termination": {
            "reason": "early_stop",
            "failed_iteration": None,
            "early_stop": {
                "enabled": True,
                "eligible": True,
                "should_stop": True,
                "reason": "gap_threshold_reached",
                "threshold_percent": 1.0,
                "system_gbest_iteration": 2,
                "system_gbest_gap_percent": 0.5,
            },
        },
        "errors": [],
    }
    _write(
        run_dir / "multi_agent2" / "multi_agent2_results.json",
        result,
    )

    analysis = build_analysis(
        run_dir=run_dir,
        manifest=_manifest(),
    )

    section = analysis["methods"]["multi_agent_2"]
    assert section["status"] == "completed"
    assert section["completed_iterations"] == 2
    assert section["termination"]["reason"] == "early_stop"
    assert section["termination"]["early_stop"][
        "system_gbest_gap_percent"
    ] == pytest.approx(0.5)


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
