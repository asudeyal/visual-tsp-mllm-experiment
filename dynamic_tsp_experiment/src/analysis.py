"""Ayrıntılı deney JSON'larından tek ve kompakt karşılaştırma raporu üretir."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core import read_json
from src.metrics import utc_now_iso


ANALYSIS_SCHEMA_VERSION = "2.0"


def _load_optional(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _status(
    result: dict[str, Any] | None,
    *,
    iterative: bool = False,
) -> str:
    if result is None:
        return "not_run"
    if result.get("schema_version") != "2.0":
        return "legacy_or_incompatible"
    if iterative:
        requested = int(result.get("requested_iterations", 0))
        completed = int(result.get("completed_iterations", 0))
        termination = result.get("termination") or {}
        if (
            termination.get("reason") == "early_stop"
            and result.get("pending_iteration") is None
        ):
            return "completed"
        if completed == requested and result.get("pending_iteration") is None:
            return "completed"
        return "partial"
    return "completed" if not result.get("errors") else "failed"


def _assert_identity(
    result: dict[str, Any] | None,
    *,
    run_id: str,
    fingerprint: str,
    path: Path,
) -> None:
    if result is None or result.get("schema_version") != "2.0":
        return
    if result.get("run_id") != run_id:
        raise ValueError(
            f"Run ID uyuşmazlığı: {path}"
        )
    actual = result.get("problem", {}).get(
        "fingerprint_sha256"
    )
    if actual != fingerprint:
        raise ValueError(
            f"Problem fingerprint uyuşmazlığı: {path}"
        )


def _compact_solution(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    validation = value.get("validation") or {}
    return {
        "source": value.get("source"),
        "iteration": value.get("iteration"),
        "candidate_id": value.get("candidate_id"),
        "is_valid": _first_not_none(
            validation.get("is_valid"),
            value.get("is_valid"),
        ),
        "distance": value.get("distance"),
        "reference_distance": value.get("reference_distance"),
        "gap_to_reference_percent": value.get(
            "gap_to_reference_percent"
        ),
    }


def _usage_tokens(call: dict[str, Any] | None) -> int:
    if not isinstance(call, dict):
        return 0
    usage = call.get("usage") or {}
    return int(usage.get("total_token_count") or 0)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _sum_numeric(
    values: list[Any],
) -> float:
    return sum(
        float(value)
        for value in values
        if value is not None
    )


def _resource_metric(
    metrics: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    value = metrics.get(name)
    if not isinstance(value, dict) or not value.get("sample_count"):
        return None
    return {
        "average": value.get("average"),
        "maximum": value.get("maximum"),
    }


def _resource_metrics(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    names = (
        "system_cpu_percent",
        "process_cpu_percent",
        "process_memory_rss_mb",
        "system_memory_percent",
        "system_memory_available_mb",
        "local_gpu_utilization_percent",
        "local_gpu_memory_used_mb",
        "local_gpu_memory_percent",
        "local_gpu_temperature_celsius",
    )
    return {
        name: compact
        for name in names
        if (compact := _resource_metric(metrics, name)) is not None
    }


def _compact_observability(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    observability = (result or {}).get("observability")
    if not isinstance(observability, dict):
        return None

    settings = observability.get("settings") or {}
    request = observability.get("request_control") or {}
    waits = (
        request.get("waits")
        or observability.get("controlled_waits")
        or {}
    )
    resources = observability.get("resources") or {}
    system = resources.get("system") or {}
    local_gpu = system.get("local_gpu") or {}
    overall = resources.get("overall") or {}

    phases: dict[str, Any] = {}
    for phase_name, phase in sorted(
        (resources.get("by_phase") or {}).items()
    ):
        phase_metrics = _resource_metrics(
            (phase or {}).get("metrics") or {}
        )
        phases[str(phase_name)] = {
            "sample_count": (phase or {}).get("sample_count"),
            "metrics": phase_metrics,
        }

    return {
        "settings": {
            "resource_profiling_enabled": settings.get(
                "profile_resources"
            ),
            "resource_sample_interval_seconds": settings.get(
                "resource_sample_interval_seconds"
            ),
            "minimum_request_interval_seconds": settings.get(
                "minimum_request_interval_seconds"
            ),
            "max_retries": settings.get("max_retries"),
            "early_stop_enabled": settings.get(
                "early_stop_enabled"
            ),
            "early_stop_gap_percent": settings.get(
                "early_stop_gap_percent"
            ),
        },
        "request_control": {
            "execution_count": request.get("execution_count"),
            "request_attempt_count": request.get(
                "request_attempt_count"
            ),
            "retry_count": request.get("retry_count"),
            "api_active_wall_seconds": request.get(
                "active_wall_seconds"
            ),
            "deliberate_delay_seconds": waits.get(
                "deliberate_delay_seconds"
            ),
            "rate_limit_backoff_seconds": waits.get(
                "rate_limit_backoff_seconds"
            ),
            "controlled_wait_seconds": waits.get(
                "controlled_wait_seconds"
            ),
            "request_total_wall_seconds": request.get(
                "total_wall_seconds"
            ),
        },
        "resources": {
            "enabled": resources.get("enabled"),
            "reason": resources.get("reason"),
            "duration_seconds": resources.get("duration_seconds"),
            "sample_count": resources.get("sample_count"),
            "local_gpu": {
                "available": local_gpu.get("available"),
                "backend": local_gpu.get("backend"),
                "unavailable_reason": local_gpu.get(
                    "unavailable_reason"
                ),
            },
            "overall_metrics": _resource_metrics(
                overall.get("metrics") or {}
            ),
            "phases": phases,
            "sampling_error_count": len(
                resources.get("sampling_errors") or []
            ),
        },
    }


def _compact_termination(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    termination = (result or {}).get("termination")
    if not isinstance(termination, dict):
        return None
    early = termination.get("early_stop")
    compact_early = None
    if isinstance(early, dict):
        compact_early = {
            "enabled": early.get("enabled"),
            "eligible": early.get("eligible"),
            "should_stop": early.get("should_stop"),
            "reason": early.get("reason"),
            "threshold_percent": early.get("threshold_percent"),
            "system_gbest_iteration": early.get(
                "system_gbest_iteration"
            ),
            "system_gbest_gap_percent": early.get(
                "system_gbest_gap_percent"
            ),
        }
    return {
        "reason": termination.get("reason"),
        "failed_iteration": termination.get("failed_iteration"),
        "early_stop": compact_early,
    }


def _compact_progress(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    progress = (result or {}).get("solution_progress")
    if not isinstance(progress, dict):
        return None
    return {
        "system_gbest": _compact_solution(
            progress.get("system_gbest")
        ),
        "observed_candidate_gbest": _compact_solution(
            progress.get("observed_candidate_gbest")
        ),
        "latest_early_stop": progress.get("latest_early_stop"),
    }


def _compact_error(error: dict[str, Any]) -> dict[str, Any]:
    call = error.get("api_call") or {}
    return {
        "phase": error.get("phase"),
        "iteration": error.get("iteration"),
        "error_type": error.get("type") or error.get("error_type"),
        "message": error.get("message"),
        "api_call_success": call.get("success"),
        "api_call_wall_seconds": call.get(
            "api_call_wall_seconds"
        ),
        "failed_stage_wall_seconds": error.get(
            "failed_stage_wall_seconds"
        ),
    }


def _baseline_section(
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    status = _status(result)
    if status != "completed" or result is None:
        return {"status": status}
    timing = result.get("timing") or {}
    return {
        "status": status,
        "or_tools": _compact_solution(result.get("or_tools")),
        "timing_seconds": {
            "problem_loading": timing.get(
                "problem_loading_seconds"
            ),
            "or_tools": timing.get("or_tools_wall_seconds"),
            "rendering": timing.get("route_rendering_seconds"),
            "total": timing.get(
                "total_wall_seconds_before_result_write"
            ),
        },
        "observability": _compact_observability(result),
    }


def _zero_shot_section(
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    status = _status(result)
    if status != "completed" or result is None:
        return {
            "status": status,
            "observability": _compact_observability(result),
            "errors": [
                _compact_error(item)
                for item in (result or {}).get("errors", [])
            ],
        }
    timing = result.get("timing") or {}
    summary = result.get("run_summary") or {}
    observability = _compact_observability(result)
    request = (observability or {}).get("request_control") or {}
    total_seconds = timing.get(
        "total_wall_seconds_before_result_write"
    )
    api_active = _first_not_none(
        timing.get("api_active_wall_seconds"),
        request.get("api_active_wall_seconds"),
        timing.get("api_call_wall_seconds"),
    )
    controlled_wait = _first_not_none(
        timing.get("controlled_wait_seconds"),
        request.get("controlled_wait_seconds"),
        0.0,
    )
    local_active = None
    if total_seconds is not None:
        local_active = max(
            0.0,
            float(total_seconds)
            - float(api_active or 0.0)
            - float(controlled_wait or 0.0),
        )
    return {
        "status": status,
        "model": result.get("model"),
        "solution": _compact_solution(result),
        "timing_seconds": {
            "api": timing.get("api_call_wall_seconds"),
            "api_active": api_active,
            "local_processing_active": local_active,
            "active": (
                None
                if local_active is None
                else local_active + float(api_active or 0.0)
            ),
            "deliberate_delay": request.get(
                "deliberate_delay_seconds"
            ),
            "rate_limit_backoff": request.get(
                "rate_limit_backoff_seconds"
            ),
            "controlled_wait": controlled_wait,
            "parsing": timing.get("response_parsing_seconds"),
            "evaluation": timing.get(
                "validation_and_metrics_seconds"
            ),
            "rendering": timing.get("route_rendering_seconds"),
            "total": total_seconds,
        },
        "api_call_count": summary.get("api_call_count"),
        "total_token_count": summary.get("total_token_count"),
        "observability": observability,
        "errors": [
            _compact_error(item)
            for item in result.get("errors", [])
        ],
    }


def _ma2_iterations(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    initializer = result.get("initializer") or {}
    initializer_validation = initializer.get("validation") or {}
    running_gbest = (
        initializer.get("distance")
        if initializer_validation.get("is_valid") is True
        else None
    )
    running_gbest_gap = (
        initializer.get("gap_to_reference_percent")
        if running_gbest is not None
        else None
    )
    for item in result.get("iterations", []):
        timing = item.get("timing") or {}
        call = item.get("api_call")
        validation = item.get("validation") or {}
        if (
            validation.get("is_valid") is True
            and item.get("distance") is not None
        ):
            if (
                running_gbest is None
                or float(item["distance"]) < float(running_gbest)
            ):
                running_gbest = item["distance"]
                running_gbest_gap = item.get(
                    "gap_to_reference_percent"
                )
        api_active = _first_not_none(
            timing.get("api_active_wall_seconds"),
            timing.get("api_call_wall_seconds"),
        )
        active = _first_not_none(
            timing.get("iteration_active_wall_seconds"),
            timing.get("iteration_total_wall_seconds"),
        )
        local_active = None
        if active is not None:
            local_active = max(
                0.0,
                float(active) - float(api_active or 0.0),
            )
        compact.append(
            {
                "iteration": item.get("iteration"),
                "status": item.get("status", "completed"),
                "is_valid": validation.get("is_valid"),
                "distance": item.get("distance"),
                "gap_to_reference_percent": item.get(
                    "gap_to_reference_percent"
                ),
                "iteration_best_distance": _first_not_none(
                    item.get("iteration_best_distance"),
                    item.get("distance"),
                ),
                "system_gbest_distance": _first_not_none(
                    item.get("system_gbest_distance"),
                    running_gbest,
                ),
                "system_gbest_gap_percent": _first_not_none(
                    item.get("system_gbest_gap_percent"),
                    running_gbest_gap,
                ),
                "timing_seconds": {
                    "api": timing.get(
                        "api_call_wall_seconds"
                    ),
                    "api_active": api_active,
                    "local_processing_active": local_active,
                    "active": active,
                    "deliberate_delay": timing.get(
                        "deliberate_delay_seconds"
                    ),
                    "rate_limit_backoff": timing.get(
                        "rate_limit_backoff_seconds"
                    ),
                    "controlled_wait": timing.get(
                        "controlled_wait_seconds"
                    ),
                    "parsing": timing.get(
                        "response_parsing_seconds"
                    ),
                    "evaluation": timing.get(
                        "validation_and_metrics_seconds"
                    ),
                    "rendering": timing.get(
                        "route_rendering_seconds"
                    ),
                    "checkpoint": timing.get(
                        "checkpoint_write_seconds"
                    ),
                    "total": timing.get(
                        "iteration_total_wall_seconds"
                    ),
                },
                "total_token_count": _first_not_none(
                    item.get("token_count"),
                    _usage_tokens(call),
                ),
            }
        )
    return compact


def _multi_agent2_section(
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    status = _status(result, iterative=True)
    if result is None or status == "legacy_or_incompatible":
        return {"status": status}
    iterations = _ma2_iterations(result)
    valid = [
        item
        for item in iterations
        if item["is_valid"] is True
    ]
    best_iteration = (
        min(
            valid,
            key=lambda item: item["distance"],
        )["iteration"]
        if valid
        else None
    )
    timing_totals = {
        key: _sum_numeric(
            [
                item.get("timing_seconds", {}).get(key)
                for item in iterations
            ]
        )
        for key in (
            "api_active",
            "local_processing_active",
            "active",
            "deliberate_delay",
            "rate_limit_backoff",
            "controlled_wait",
            "total",
        )
    }
    return {
        "status": status,
        "model": result.get("model"),
        "requested_iterations": result.get(
            "requested_iterations"
        ),
        "completed_iterations": result.get(
            "completed_iterations"
        ),
        "valid_iteration_count": len(valid),
        "invalid_iteration_count": (
            len(iterations) - len(valid)
        ),
        "best_valid_iteration": best_iteration,
        "initializer": _compact_solution(
            result.get("initializer")
        ),
        "final_solution": _compact_solution(
            result.get("final_solution")
        ),
        "best_valid_solution": _compact_solution(
            result.get("best_valid_solution")
        ),
        "solution_progress": _compact_progress(result),
        "termination": _compact_termination(result),
        "iterations": iterations,
        "timing_seconds": timing_totals,
        "total_token_count": sum(
            int(item.get("total_token_count") or 0)
            for item in iterations
        ),
        "observability": _compact_observability(result),
        "run_summary": result.get("run_summary"),
        "errors": [
            _compact_error(item)
            for item in result.get("errors", [])
        ],
    }


def _ma1_iterations(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    initializer = result.get("initializer") or {}
    initializer_validation = initializer.get("validation") or {}
    running_system_gbest = (
        initializer.get("distance")
        if initializer_validation.get("is_valid") is True
        else None
    )
    running_system_gbest_gap = (
        initializer.get("gap_to_reference_percent")
        if running_system_gbest is not None
        else None
    )
    running_observed_gbest = None
    for item in result.get("iterations", []):
        critic = item.get("critic") or {}
        scorer = item.get("scorer") or {}
        candidates = critic.get("candidates") or []
        valid_candidates = [
            candidate
            for candidate in candidates
            if (
                candidate.get("validation", {}).get("is_valid")
                is True
            )
        ]
        valid_distances = [
            candidate["distance"]
            for candidate in valid_candidates
            if candidate.get("distance") is not None
        ]
        best_valid_distance = (
            min(valid_distances)
            if valid_distances
            else None
        )
        selected = item.get("selected_solution") or {}
        selected_validation = selected.get("validation") or {}
        selected_distance = selected.get("distance")
        if best_valid_distance is not None:
            running_observed_gbest = min(
                float(running_observed_gbest),
                float(best_valid_distance),
            ) if running_observed_gbest is not None else best_valid_distance
        if (
            selected_validation.get("is_valid") is True
            and selected_distance is not None
        ):
            if (
                running_system_gbest is None
                or float(selected_distance)
                < float(running_system_gbest)
            ):
                running_system_gbest = selected_distance
                running_system_gbest_gap = selected.get(
                    "gap_to_reference_percent"
                )
        selection_regret = _first_not_none(
            item.get("selection_regret_percent"),
            scorer.get(
                "selection_regret_percent_after_evaluation"
            ),
        )
        if (
            selection_regret is None
            and best_valid_distance is not None
            and selected_distance is not None
            and best_valid_distance != 0
        ):
            selection_regret = (
                100.0
                * (selected_distance - best_valid_distance)
                / best_valid_distance
            )
        selected_best = _first_not_none(
            item.get("selected_is_iteration_best"),
            (
                abs(float(selection_regret)) <= 1e-9
                if selection_regret is not None
                else None
            ),
        )
        timing = item.get("timing") or {}
        critic_timing = critic.get("timing") or {}
        scorer_timing = scorer.get("timing") or {}
        critic_tokens = _first_not_none(
            critic.get("token_count"),
            _usage_tokens(critic.get("api_call")),
        )
        scorer_tokens = _first_not_none(
            scorer.get("token_count"),
            _usage_tokens(scorer.get("api_call")),
        )
        total_tokens = _first_not_none(
            item.get("token_count"),
            int(critic_tokens or 0) + int(scorer_tokens or 0),
        )
        active = _first_not_none(
            timing.get("iteration_active_wall_seconds"),
            timing.get("iteration_processing_wall_seconds"),
        )
        api_active = _sum_numeric(
            [
                _first_not_none(
                    critic_timing.get("api_active_wall_seconds"),
                    critic_timing.get("api_call_wall_seconds"),
                ),
                _first_not_none(
                    scorer_timing.get("api_active_wall_seconds"),
                    scorer_timing.get("api_call_wall_seconds"),
                ),
            ]
        )
        local_active = _first_not_none(
            timing.get("local_processing_active_wall_seconds"),
            (
                max(0.0, float(active) - api_active)
                if active is not None
                else None
            ),
        )
        compact.append(
            {
                "iteration": item.get("iteration"),
                "returned_candidate_count": critic.get(
                    "returned_candidate_count"
                ),
                "valid_candidate_count": len(valid_candidates),
                "invalid_candidate_count": (
                    len(candidates) - len(valid_candidates)
                ),
                "best_valid_candidate_distance": (
                    best_valid_distance
                ),
                "iteration_best_distance": _first_not_none(
                    item.get("iteration_best_distance"),
                    best_valid_distance,
                ),
                "system_gbest_distance": _first_not_none(
                    item.get("system_gbest_distance"),
                    running_system_gbest,
                ),
                "system_gbest_gap_percent": _first_not_none(
                    item.get("system_gbest_gap_percent"),
                    running_system_gbest_gap,
                ),
                "observed_candidate_gbest_distance": _first_not_none(
                    item.get("observed_candidate_gbest_distance"),
                    running_observed_gbest,
                ),
                "selection_mode": scorer.get("selection_mode"),
                "selected_candidate_id": scorer.get(
                    "best_candidate_id"
                ),
                "selected_is_valid": selected_validation.get(
                    "is_valid"
                ),
                "selected_distance": selected_distance,
                "selected_gap_to_reference_percent": selected.get(
                    "gap_to_reference_percent"
                ),
                "selection_regret_percent": selection_regret,
                "selected_best_valid_candidate": selected_best,
                "timing_seconds": {
                    "critic_api": critic_timing.get(
                        "api_call_wall_seconds"
                    ),
                    "critic_total": timing.get(
                        "critic_stage_wall_seconds"
                    ),
                    "scorer_api": scorer_timing.get(
                        "api_call_wall_seconds"
                    ),
                    "scorer_total": timing.get(
                        "scorer_stage_wall_seconds"
                    ),
                    "api_active": api_active,
                    "local_processing_active": local_active,
                    "active": active,
                    "deliberate_delay": timing.get(
                        "deliberate_delay_seconds"
                    ),
                    "rate_limit_backoff": timing.get(
                        "rate_limit_backoff_seconds"
                    ),
                    "controlled_wait": timing.get(
                        "controlled_wait_seconds"
                    ),
                    "checkpoint": timing.get(
                        "checkpoint_write_seconds"
                    ),
                    "total": timing.get(
                        "iteration_observed_total_wall_seconds",
                        timing.get("iteration_processing_wall_seconds"),
                    ),
                },
                "token_count": {
                    "critic": critic_tokens,
                    "scorer": scorer_tokens,
                    "total": total_tokens,
                },
            }
        )
    return compact


def _multi_agent1_section(
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    status = _status(result, iterative=True)
    if result is None or status == "legacy_or_incompatible":
        return {"status": status}
    iterations = _ma1_iterations(result)
    total_candidates = sum(
        int(item["returned_candidate_count"] or 0)
        for item in iterations
    )
    valid_candidates = sum(
        item["valid_candidate_count"]
        for item in iterations
    )
    scorer_iterations = [
        item
        for item in iterations
        if item["selection_mode"]
        == "visual_scorer_after_feasibility_filter"
    ]
    scorer_best = sum(
        item["selected_best_valid_candidate"] is True
        for item in scorer_iterations
    )
    timing_totals = {
        key: _sum_numeric(
            [
                item.get("timing_seconds", {}).get(key)
                for item in iterations
            ]
        )
        for key in (
            "api_active",
            "local_processing_active",
            "active",
            "deliberate_delay",
            "rate_limit_backoff",
            "controlled_wait",
            "total",
        )
    }
    return {
        "status": status,
        "model": result.get("model"),
        "candidate_count_requested": result.get(
            "candidate_count_requested"
        ),
        "requested_iterations": result.get(
            "requested_iterations"
        ),
        "completed_iterations": result.get(
            "completed_iterations"
        ),
        "total_candidate_count": total_candidates,
        "valid_candidate_count": valid_candidates,
        "invalid_candidate_count": (
            total_candidates - valid_candidates
        ),
        "valid_candidate_rate_percent": (
            100.0 * valid_candidates / total_candidates
            if total_candidates
            else None
        ),
        "scorer_evaluated_iteration_count": len(
            scorer_iterations
        ),
        "scorer_best_candidate_selection_count": scorer_best,
        "scorer_best_candidate_selection_rate_percent": (
            100.0 * scorer_best / len(scorer_iterations)
            if scorer_iterations
            else None
        ),
        "initializer": _compact_solution(
            result.get("initializer")
        ),
        "final_solution": _compact_solution(
            result.get("final_solution")
        ),
        "best_valid_solution": _compact_solution(
            result.get("best_valid_solution")
        ),
        "best_critic_candidate_oracle": _compact_solution(
            result.get("best_critic_candidate_oracle")
        ),
        "solution_progress": _compact_progress(result),
        "termination": _compact_termination(result),
        "iterations": iterations,
        "timing_seconds": timing_totals,
        "total_token_count": sum(
            int((item.get("token_count") or {}).get("total") or 0)
            for item in iterations
        ),
        "observability": _compact_observability(result),
        "run_summary": result.get("run_summary"),
        "errors": [
            _compact_error(item)
            for item in result.get("errors", [])
        ],
    }


def _ranking_entry(
    method: str,
    solution: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if (
        solution is None
        or solution.get("is_valid") is not True
        or solution.get("distance") is None
    ):
        return None
    return {
        "method": method,
        "distance": solution["distance"],
        "gap_to_reference_percent": solution.get(
            "gap_to_reference_percent"
        ),
    }


def _model_identity(
    result: dict[str, Any] | None,
    *,
    fallback_provider: str,
    fallback_alias: str,
) -> tuple[str, str, str | None]:
    model = (result or {}).get("model")
    if isinstance(model, str):
        return fallback_provider, fallback_alias or model, model
    model = model or {}
    provider = str(
        model.get("provider") or fallback_provider
    )
    if provider == "google_gemini":
        provider = "gemini"
    alias = str(
        model.get("alias")
        or model.get("name")
        or fallback_alias
    )
    resolved = (
        model.get("requested_name")
        or model.get("name")
        or alias
    )
    return provider, alias, str(resolved) if resolved else None


def _provider_result_sets(
    run_dir: Path,
    *,
    legacy_results: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Yeni provider düzeniyle eski Gemini/OpenRouter düzenini keşfeder."""

    discovered: list[dict[str, Any]] = []

    # Eski dinamik Gemini sonuçları.
    legacy_anchor = next(
        (
            legacy_results[name]
            for name in (
                "zero_shot",
                "multi_agent_1",
                "multi_agent_2",
            )
            if legacy_results.get(name) is not None
        ),
        None,
    )
    if legacy_anchor is not None:
        provider, alias, resolved = _model_identity(
            legacy_anchor,
            fallback_provider="gemini",
            fallback_alias="gemini",
        )
        discovered.append(
            {
                "provider": provider,
                "model_alias": alias,
                "resolved_model": resolved,
                "layout": "legacy_gemini",
                "paths": {
                    "zero_shot": (
                        run_dir
                        / "zero_shot"
                        / "zero_shot_results.json"
                    ),
                    "multi_agent_1": (
                        run_dir
                        / "multi_agent1"
                        / "multi_agent1_results.json"
                    ),
                    "multi_agent_2": (
                        run_dir
                        / "multi_agent2"
                        / "multi_agent2_results.json"
                    ),
                },
            }
        )

    # Eski OpenRouter karşılaştırma sonuçları.
    openrouter_root = (
        run_dir / "model_comparisons" / "openrouter"
    )
    if openrouter_root.exists():
        for model_dir in sorted(
            path
            for path in openrouter_root.iterdir()
            if path.is_dir()
        ):
            paths = {
                "zero_shot": (
                    model_dir / "zero_shot_results.json"
                ),
                "multi_agent_1": (
                    model_dir
                    / "multi_agent1"
                    / "multi_agent1_results.json"
                ),
                "multi_agent_2": (
                    model_dir
                    / "multi_agent2"
                    / "multi_agent2_results.json"
                ),
            }
            anchor = next(
                (
                    _load_optional(path)
                    for path in paths.values()
                    if path.exists()
                ),
                None,
            )
            provider, alias, resolved = _model_identity(
                anchor,
                fallback_provider="openrouter",
                fallback_alias=model_dir.name,
            )
            discovered.append(
                {
                    "provider": provider,
                    "model_alias": alias,
                    "resolved_model": resolved,
                    "layout": "legacy_openrouter",
                    "paths": paths,
                }
            )

    # Yeni ortak provider/model/method düzeni.
    providers_root = run_dir / "providers"
    if providers_root.exists():
        for provider_dir in sorted(
            path
            for path in providers_root.iterdir()
            if path.is_dir()
        ):
            for model_dir in sorted(
                path
                for path in provider_dir.iterdir()
                if path.is_dir()
            ):
                paths = {
                    "zero_shot": (
                        model_dir
                        / "zero_shot"
                        / "zero_shot_results.json"
                    ),
                    "multi_agent_1": (
                        model_dir
                        / "multi_agent1"
                        / "multi_agent1_results.json"
                    ),
                    "multi_agent_2": (
                        model_dir
                        / "multi_agent2"
                        / "multi_agent2_results.json"
                    ),
                }
                anchor = next(
                    (
                        _load_optional(path)
                        for path in paths.values()
                        if path.exists()
                    ),
                    None,
                )
                provider, alias, resolved = _model_identity(
                    anchor,
                    fallback_provider=provider_dir.name,
                    fallback_alias=model_dir.name,
                )
                discovered.append(
                    {
                        "provider": provider,
                        "model_alias": alias,
                        "resolved_model": resolved,
                        "layout": "unified_provider",
                        "paths": paths,
                    }
                )
    return discovered


def _provider_models_section(
    *,
    run_dir: Path,
    run_id: str,
    fingerprint: str,
    legacy_results: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for discovered in _provider_result_sets(
        run_dir,
        legacy_results=legacy_results,
    ):
        key = (
            discovered["provider"],
            discovered["model_alias"],
        )
        current = merged.get(key)
        if current is None:
            merged[key] = {
                **discovered,
                "paths": dict(discovered["paths"]),
            }
            continue
        for method, path in discovered["paths"].items():
            if path.exists():
                current["paths"][method] = path
        current["resolved_model"] = (
            discovered["resolved_model"]
            or current["resolved_model"]
        )
        current["layout"] = "mixed_legacy_and_unified"

    for discovered in sorted(
        merged.values(),
        key=lambda item: (
            item["provider"],
            item["model_alias"],
        ),
    ):
        paths = discovered["paths"]
        results = {
            name: _load_optional(path)
            for name, path in paths.items()
        }
        for name, result in results.items():
            _assert_identity(
                result,
                run_id=run_id,
                fingerprint=fingerprint,
                path=paths[name],
            )
        zero = _zero_shot_section(results["zero_shot"])
        ma1 = _multi_agent1_section(results["multi_agent_1"])
        ma2 = _multi_agent2_section(results["multi_agent_2"])
        models.append(
            {
                "provider": discovered["provider"],
                "model_alias": discovered["model_alias"],
                "resolved_model": discovered["resolved_model"],
                "layout": discovered["layout"],
                "methods": {
                    "zero_shot": zero,
                    "multi_agent_1": ma1,
                    "multi_agent_2": ma2,
                },
            }
        )
    return models


def _provider_comparison_rows(
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        methods = model["methods"]
        values = (
            (
                "zero_shot",
                methods["zero_shot"],
                methods["zero_shot"].get("solution"),
            ),
            (
                "multi_agent_1",
                methods["multi_agent_1"],
                methods["multi_agent_1"].get(
                    "best_valid_solution"
                ),
            ),
            (
                "multi_agent_2",
                methods["multi_agent_2"],
                methods["multi_agent_2"].get(
                    "best_valid_solution"
                ),
            ),
        )
        for method_name, section, solution in values:
            run_summary = section.get("run_summary") or {}
            timing = section.get("timing_seconds") or {}
            observability = section.get("observability") or {}
            request = observability.get("request_control") or {}
            resources = observability.get("resources") or {}
            rows.append(
                {
                    "provider": model["provider"],
                    "model_alias": model["model_alias"],
                    "method": method_name,
                    "status": section.get("status"),
                    "completed_iterations": section.get(
                        "completed_iterations"
                    ),
                    "is_valid": (solution or {}).get("is_valid"),
                    "distance": (solution or {}).get("distance"),
                    "gap_to_reference_percent": (
                        solution or {}
                    ).get("gap_to_reference_percent"),
                    "api_call_count": (
                        section.get("api_call_count")
                        or run_summary.get("api_call_count")
                    ),
                    "api_wall_seconds": (
                        _first_not_none(
                            timing.get("api_active"),
                            timing.get("api"),
                            request.get("api_active_wall_seconds"),
                            run_summary.get(
                                "total_api_call_wall_seconds"
                            ),
                        )
                    ),
                    "total_token_count": (
                        _first_not_none(
                            section.get("total_token_count"),
                            run_summary.get("total_token_count"),
                        )
                    ),
                    "active_wall_seconds": _first_not_none(
                        timing.get("active"),
                        request.get("api_active_wall_seconds"),
                    ),
                    "deliberate_delay_seconds": _first_not_none(
                        timing.get("deliberate_delay"),
                        request.get("deliberate_delay_seconds"),
                    ),
                    "rate_limit_backoff_seconds": _first_not_none(
                        timing.get("rate_limit_backoff"),
                        request.get("rate_limit_backoff_seconds"),
                    ),
                    "controlled_wait_seconds": _first_not_none(
                        timing.get("controlled_wait"),
                        request.get("controlled_wait_seconds"),
                    ),
                    "total_wall_seconds": timing.get("total"),
                    "retry_count": request.get("retry_count"),
                    "resource_profile_enabled": resources.get(
                        "enabled"
                    ),
                    "error_count": len(section.get("errors", [])),
                }
            )
    return rows


def build_analysis(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    run_id = manifest["run_id"]
    problem = manifest["problem"]
    fingerprint = problem["fingerprint_sha256"]
    paths = {
        "baseline": run_dir / "baseline" / "baseline_results.json",
        "zero_shot": run_dir / "zero_shot" / "zero_shot_results.json",
        "multi_agent_1": (
            run_dir
            / "multi_agent1"
            / "multi_agent1_results.json"
        ),
        "multi_agent_2": (
            run_dir
            / "multi_agent2"
            / "multi_agent2_results.json"
        ),
    }
    results = {
        name: _load_optional(path)
        for name, path in paths.items()
    }
    for name, result in results.items():
        _assert_identity(
            result,
            run_id=run_id,
            fingerprint=fingerprint,
            path=paths[name],
        )

    baseline = _baseline_section(results["baseline"])
    zero = _zero_shot_section(results["zero_shot"])
    ma1 = _multi_agent1_section(results["multi_agent_1"])
    ma2 = _multi_agent2_section(results["multi_agent_2"])
    provider_models = _provider_models_section(
        run_dir=run_dir,
        run_id=run_id,
        fingerprint=fingerprint,
        legacy_results=results,
    )
    provider_rows = _provider_comparison_rows(provider_models)
    valid_provider_rows = [
        row
        for row in provider_rows
        if (
            row["is_valid"] is True
            and row["distance"] is not None
        )
    ]

    candidates = [
        _ranking_entry("or_tools", baseline.get("or_tools")),
        _ranking_entry("zero_shot", zero.get("solution")),
        _ranking_entry(
            "multi_agent_1_best_valid",
            ma1.get("best_valid_solution"),
        ),
        _ranking_entry(
            "multi_agent_2_best_valid",
            ma2.get("best_valid_solution"),
        ),
    ]
    ranking = sorted(
        [
            item
            for item in candidates
            if item is not None
        ],
        key=lambda item: item["distance"],
    )
    mllm_ranking = [
        item
        for item in ranking
        if item["method"] != "or_tools"
    ]
    statuses = {
        "baseline": baseline["status"],
        "zero_shot": zero["status"],
        "multi_agent_1": ma1["status"],
        "multi_agent_2": ma2["status"],
    }
    provider_completion = [
        {
            "provider": model["provider"],
            "model_alias": model["model_alias"],
            "methods": {
                name: section["status"]
                for name, section in model["methods"].items()
            },
            "all_methods_completed": all(
                section["status"] == "completed"
                for section in model["methods"].values()
            ),
        }
        for model in provider_models
    ]
    all_provider_models_completed = bool(provider_completion) and all(
        item["all_methods_completed"]
        for item in provider_completion
    )
    legacy_all_completed = all(
        value == "completed"
        for value in statuses.values()
    )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment": "dynamic_tsp_comparative_analysis",
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "problem": {
            "name": problem["name"],
            "source_type": problem["source_type"],
            "dimension": problem["dimension"],
            "depot_id": problem["depot_id"],
            "edge_weight_type": problem["edge_weight_type"],
            "fingerprint_sha256": fingerprint,
            "reference": {
                "type": (
                    problem.get("reference") or {}
                ).get("type"),
                "distance": (
                    problem.get("reference") or {}
                ).get("distance"),
                "is_proven_optimal": (
                    problem.get("reference") or {}
                ).get("is_proven_optimal"),
            },
        },
        "measurement_definitions": {
            "active_seconds": (
                "Kontrollü beklemeler çıkarıldıktan sonra yerel işlem "
                "ve API isteğinde geçen duvar saati süresi."
            ),
            "api_active_seconds": (
                "İstek denemelerinin istemci tarafından gözlenen "
                "süresi; ağ, sağlayıcı kuyruğu ve model çıkarımı "
                "ayrı ayrı ölçülemez."
            ),
            "deliberate_delay_seconds": (
                "İstek aralığı kuralı nedeniyle bilinçli bekleme."
            ),
            "rate_limit_backoff_seconds": (
                "429/503/504 sonrasında kontrollü retry beklemesi."
            ),
            "system_gbest": (
                "Yöntemin seçilmiş ve sonraki iterasyona aktardığı "
                "çözümler arasındaki en iyi geçerli mesafe."
            ),
            "observed_candidate_gbest": (
                "Multi-Agent 1 critic adayları içinde şimdiye kadar "
                "gözlenen en iyi geçerli mesafe."
            ),
        },
        "completion": {
            "methods": statuses,
            "provider_models": provider_completion,
            "all_provider_models_completed": (
                all_provider_models_completed
            ),
            "all_methods_completed": (
                legacy_all_completed
                or all_provider_models_completed
            ),
        },
        "methods": {
            "baseline": baseline,
            "zero_shot": zero,
            "multi_agent_1": ma1,
            "multi_agent_2": ma2,
        },
        "provider_models": provider_models,
        "comparison": {
            "representative_solution_policy": {
                "baseline": "or_tools",
                "zero_shot": "single_solution",
                "multi_agent_1": "best_valid_scorer_selection",
                "multi_agent_2": "best_valid_critic_iteration",
            },
            "method_ranking_by_valid_distance": ranking,
            "best_valid_mllm_solution": (
                mllm_ranking[0]
                if mllm_ranking
                else None
            ),
            "all_model_method_rows": provider_rows,
            "best_valid_provider_model_method": (
                min(
                    valid_provider_rows,
                    key=lambda item: item["distance"],
                )
                if valid_provider_rows
                else None
            ),
        },
    }
