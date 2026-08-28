"""Compact AVMA-TSP analysis with protocol-aware completion semantics."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .compact import TraceStore, read_state, trace_api_metrics


_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}
_CONTEXT_KEYS = ("iteration", "candidate", "scope", "attempt", "restart_attempt")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "evet" if value else "hayır"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _compact(value: Any, maximum: int) -> str:
    text = _cell(value).replace("\r", " ").replace("\n", " ")
    if len(text) <= maximum:
        return text
    return text[: max(1, maximum - 1)] + "…"


def _render_table(
    title: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    right_align: set[int] | None = None,
    max_widths: dict[int, int] | None = None,
) -> str:
    values = [[_cell(value) for value in row] for row in rows]
    header_values = [str(value) for value in headers]
    if any(len(row) != len(header_values) for row in values):
        raise ValueError("Tablo satır ve başlık sütun sayıları eşit değil")

    limits = max_widths or {}
    for row in values:
        for index, value in enumerate(row):
            if index in limits:
                row[index] = _compact(value, limits[index])

    widths = [
        max(len(header_values[index]), *([len(row[index]) for row in values] or [0]))
        for index in range(len(header_values))
    ]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    aligns = right_align or set()

    def row_text(row: Sequence[str]) -> str:
        cells: list[str] = []
        for index, value in enumerate(row):
            rendered = value.rjust(widths[index]) if index in aligns else value.ljust(widths[index])
            cells.append(f" {rendered} ")
        return "│" + "│".join(cells) + "│"

    lines = [title, border("┌", "┬", "┐"), row_text(header_values), border("├", "┼", "┤")]
    if values:
        lines.extend(row_text(row) for row in values)
    else:
        empty = ["Kayıt yok.", *([""] * (len(headers) - 1))]
        lines.append(row_text(empty))
    lines.append(border("└", "┴", "┘"))
    return "\n".join(lines)


def _side_by_side(left: str, right: str, *, gap: int = 3) -> str:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    left_width = max((len(line) for line in left_lines), default=0)
    height = max(len(left_lines), len(right_lines))
    rows: list[str] = []
    for index in range(height):
        lhs = left_lines[index] if index < len(left_lines) else ""
        rhs = right_lines[index] if index < len(right_lines) else ""
        rows.append(lhs.ljust(left_width) + (" " * gap) + rhs)
    return "\n".join(rows).rstrip()


def _valid_distance(evaluation: dict[str, Any] | None) -> float | None:
    if not isinstance(evaluation, dict):
        return None
    validation = evaluation.get("validation") or {}
    distance = evaluation.get("distance")
    if validation.get("valid") is True and distance is not None:
        return float(distance)
    return None


def _gap(distance: Any, reference: Any) -> float | None:
    if distance is None or reference is None or float(reference) <= 0:
        return None
    return ((float(distance) - float(reference)) / float(reference)) * 100.0


def _iteration_from_scope(scope: Any) -> int | None:
    match = re.search(r"iteration_(\d+)", str(scope or ""))
    return int(match.group(1)) if match else None


def _event_iteration(event: dict[str, Any]) -> int | None:
    value = event.get("iteration")
    if isinstance(value, int):
        return value
    return _iteration_from_scope(event.get("scope"))


def _status_code(event: dict[str, Any]) -> int | str | None:
    code = event.get("status_code")
    if code is not None:
        try:
            return int(code)
        except (TypeError, ValueError):
            return code
    match = re.search(r"\b([45]\d{2})\b", str(event.get("message") or ""))
    return int(match.group(1)) if match else None


def _agent_from_phase(phase: Any) -> str | None:
    value = str(phase or "")
    if value.startswith("initializer"):
        return "initializer"
    if value.startswith("critic_candidate"):
        return "critic"
    if value.startswith("visual_scorer"):
        return "scorer"
    if value.startswith("repair_attempt"):
        return "repair"
    if value.startswith("diversity_restart"):
        return "diversity"
    if value.startswith("hybrid_visual_two_opt"):
        return "hybrid"
    return None


def _context_matches(error: dict[str, Any], event: dict[str, Any]) -> bool:
    for key in _CONTEXT_KEYS:
        if key in error and error.get(key) != event.get(key):
            return False
    return True


def _iteration_result_is_complete(event: dict[str, Any]) -> bool:
    result = event.get("result") or {}
    selected = result.get("selected_before_repair") or {}
    working = result.get("working_evaluation") or {}
    working_validation = working.get("validation") or {}
    candidates = result.get("critic_candidates") or []
    return (
        bool(candidates)
        and selected.get("candidate_id") is not None
        and working_validation.get("valid") is True
    )


def _provider_error_resolved(error: dict[str, Any], later_events: Sequence[dict[str, Any]]) -> bool:
    """Return True when the failed provider stage was later superseded by real progress.

    Historical traces may contain provider errors from an earlier attempt in a
    Repair/Restart chain that later succeeded. Those errors must not truncate
    the analysis. Conversely, an iteration that only emitted a bookkeeping
    ``iteration_result`` without a selected candidate is not real progress.
    """

    error_seq = int(error.get("seq") or 0)
    error_iteration = _event_iteration(error)
    agent = _agent_from_phase(error.get("phase"))

    for event in later_events:
        if int(event.get("seq") or 0) <= error_seq:
            continue

        kind = event.get("event")

        # Initializer-level provider failures are superseded only once a valid
        # initializer_result is actually persisted. This also covers a failed
        # Repair attempt followed by a later successful Repair/Restart attempt.
        if error_iteration is None and kind == "initializer_result":
            evaluation = event.get("evaluation") or {}
            if (evaluation.get("validation") or {}).get("valid") is True:
                return True

        # For search iterations, the strongest evidence of recovery is a real
        # completed iteration: candidates existed, Scorer produced a selection,
        # and the persisted working route is feasible. Old ghost iterations do
        # not satisfy this shape.
        if (
            error_iteration is not None
            and kind == "iteration_result"
            and _event_iteration(event) == error_iteration
            and _iteration_result_is_complete(event)
        ):
            return True

        # Resume may retry exactly the same stage before iteration_result exists.
        if kind == "agent_call" and agent is not None:
            if event.get("agent") == agent and _context_matches(error, event):
                return True

        if kind in {"model_output_failure", "recoverable_agent_failure"}:
            if event.get("error_type") != "ModelOutputError":
                continue
            if event.get("phase") == error.get("phase") and _context_matches(error, event):
                return True

    return False


def _first_unresolved_provider_error(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    provider_errors = [event for event in events if event.get("event") == "provider_error"]
    for error in provider_errors:
        if not _provider_error_resolved(error, events):
            return error
    return None


def _provider_wait_seconds(events: Iterable[dict[str, Any]]) -> float:
    total = 0.0
    for event in events:
        kind = event.get("event")
        if kind == "agent_call":
            metadata = ((event.get("call") or {}).get("raw_metadata") or {})
            total += float(metadata.get("provider_wait_seconds") or 0.0)
        elif kind == "model_output_attempt":
            metadata = ((event.get("provider_response") or {}).get("raw_metadata") or {})
            total += float(metadata.get("provider_wait_seconds") or 0.0)
        elif kind == "provider_error":
            total += float(event.get("provider_wait_seconds") or 0.0)
    return total


def _events_through(events: Sequence[dict[str, Any]], seq: int | None) -> list[dict[str, Any]]:
    if seq is None:
        return list(events)
    return [event for event in events if int(event.get("seq") or 0) <= seq]


def _events_for_iteration(events: Sequence[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
    return [event for event in events if _event_iteration(event) == iteration]


def _candidate_metrics(data: dict[str, Any]) -> tuple[int, int, float | None]:
    candidates = data.get("critic_candidates") or []
    distances = [
        distance
        for candidate in candidates
        if (distance := _valid_distance(candidate.get("evaluation"))) is not None
    ]
    return len(candidates), len(distances), min(distances) if distances else None


def _prompt_set(raw_config: dict[str, Any], experiment_cfg: dict[str, Any]) -> str:
    for source in (raw_config, experiment_cfg):
        value = source.get("prompt_set")
        if value:
            return str(value)
    name = str(experiment_cfg.get("name") or "")
    match = re.search(r"(?:^|_)(v\d+)$", name)
    return match.group(1) if match else "-"


def _initializer_stats(trace: TraceStore, analysis_events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    allowed = {int(event.get("seq") or 0) for event in analysis_events}

    def usable(event: dict[str, Any]) -> bool:
        return int(event.get("seq") or 0) in allowed

    initial_events = [event for event in trace.matching("initializer_candidate") if usable(event)]
    final_events = [event for event in trace.matching("initializer_result") if usable(event)]
    initial = initial_events[-1] if initial_events else {}
    final = final_events[-1] if final_events else {}
    initial_eval = initial.get("evaluation") or {}
    initial_valid = (initial_eval.get("validation") or {}).get("valid")

    direct_repairs = [
        event for event in trace.matching("repair_result")
        if usable(event) and event.get("scope") == "initializer"
    ]
    fallback = [
        event for event in trace.matching("diversity_result")
        if usable(event) and event.get("scope") == "initializer.fallback"
    ]

    repair_result = "yok"
    if direct_repairs:
        last_validation = (direct_repairs[-1].get("evaluation") or {}).get("validation") or {}
        repair_result = "başarılı" if last_validation.get("valid") is True else "başarısız"

    if initial_valid is True:
        accepted_source = "initializer"
    elif repair_result == "başarılı":
        accepted_source = "repair"
    elif fallback:
        accepted_source = "fallback restart"
    else:
        accepted_source = "-"

    accepted_eval = final.get("evaluation") or {}
    return {
        "valid_on_first_attempt": initial_valid,
        "repair_required": initial_valid is False if initial_valid is not None else None,
        "repair_attempt_count": len(direct_repairs),
        "repair_result": repair_result,
        "fallback_restart_count": len(fallback),
        "accepted_source": accepted_source,
        "accepted_distance": accepted_eval.get("distance"),
    }


def _repair_stats(events: Sequence[dict[str, Any]], completed_iterations: set[int]) -> dict[str, Any]:
    activations: dict[str, set[int]] = {}
    success_scopes: set[str] = set()

    for event in events:
        iteration = _event_iteration(event)
        if iteration not in completed_iterations:
            continue
        scope = str(event.get("scope") or "")
        if not scope.startswith("iteration_"):
            continue

        is_repair = (
            (event.get("event") == "agent_call" and event.get("agent") == "repair")
            or (event.get("event") in {"recoverable_agent_failure", "model_output_failure"}
                and _agent_from_phase(event.get("phase")) == "repair")
            or event.get("event") == "repair_result"
        )
        if not is_repair:
            continue

        attempt = int(event.get("attempt") or 0)
        activations.setdefault(scope, set()).add(attempt)
        if event.get("event") == "repair_result":
            validation = (event.get("evaluation") or {}).get("validation") or {}
            if validation.get("valid") is True:
                success_scopes.add(scope)

    activation_count = len(activations)
    success_count = len(success_scopes)
    return {
        "activation_count": activation_count,
        "successful_repair_count": success_count,
        "failed_repair_count": activation_count - success_count,
        "total_attempt_count": sum(len(attempts) for attempts in activations.values()),
    }


def _recovery_stats(
    events: Sequence[dict[str, Any]],
    completed_data: Sequence[dict[str, Any]],
    repair: dict[str, Any],
) -> dict[str, Any]:
    completed = {int(data["iteration"]) for data in completed_data}
    attempts: set[tuple[str, int]] = set()
    chain_scopes: set[str] = set()
    successful_chains: set[str] = set()

    def feasibility_scope(scope: str) -> bool:
        return scope.startswith("iteration_") and ".escape_restart" not in scope

    for event in events:
        iteration = _event_iteration(event)
        if iteration not in completed:
            continue
        scope = str(event.get("scope") or "")
        if not feasibility_scope(scope):
            continue

        is_diversity = (
            (event.get("event") == "agent_call" and event.get("agent") == "diversity")
            or (event.get("event") in {"recoverable_agent_failure", "model_output_failure"}
                and _agent_from_phase(event.get("phase")) == "diversity")
            or event.get("event") == "diversity_result"
        )
        if is_diversity and ".restart_" not in scope:
            restart_attempt = int(event.get("restart_attempt") or 0)
            attempts.add((scope, restart_attempt))
            chain_scopes.add(scope)
            if event.get("event") == "diversity_result":
                validation = (event.get("evaluation") or {}).get("validation") or {}
                if validation.get("valid") is True:
                    successful_chains.add(scope)

    # A diversity proposal may itself be repaired before being accepted.
    for event in events:
        if event.get("event") != "repair_result":
            continue
        iteration = _event_iteration(event)
        if iteration not in completed:
            continue
        repair_scope = str(event.get("scope") or "")
        validation = (event.get("evaluation") or {}).get("validation") or {}
        if validation.get("valid") is not True:
            continue
        for chain_scope in chain_scopes:
            if repair_scope.startswith(chain_scope + ".restart_"):
                successful_chains.add(chain_scope)

    exhausted = {
        str(event.get("scope") or "")
        for event in events
        if event.get("event") == "restart_exhausted"
        and _event_iteration(event) in completed
        and feasibility_scope(str(event.get("scope") or ""))
    }
    retained = {
        str(event.get("scope") or "")
        for event in events
        if event.get("event") == "restart_exhausted"
        and _event_iteration(event) in completed
        and feasibility_scope(str(event.get("scope") or ""))
        and event.get("fallback_action") == "retain_incumbent"
    }

    return {
        **repair,
        "feasibility_restart_attempt_count": len(attempts),
        "restart_rescue_count": len(successful_chains),
        "restart_exhaustion_count": len(exhausted),
        "incumbent_retained_count": len(retained),
    }


def _robustness_stats(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    model_failures = [
        event for event in events
        if event.get("event") == "recoverable_agent_failure"
        and event.get("error_type") == "ModelOutputError"
    ]
    by_agent = Counter(_agent_from_phase(event.get("phase")) or "unknown" for event in model_failures)
    provider_errors = [event for event in events if event.get("event") == "provider_error"]
    timeout_count = sum("timeout" in str(event.get("error_type") or "").lower() for event in provider_errors)
    transient_http_count = sum(_status_code(event) in _TRANSIENT_HTTP for event in provider_errors)
    parse_failures = sum(
        event.get("event") == "model_output_failure"
        for event in events
    )
    return {
        "recoverable_model_failure_count": len(model_failures),
        "by_agent": dict(by_agent),
        "parse_failure_count": parse_failures,
        "timeout_count": timeout_count,
        "transient_http_count": transient_http_count,
    }


def _adaptive_stats(completed_data: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stagnation = 0
    hybrid = 0
    valid_two_opt = 0
    restart = 0
    for data in completed_data:
        if (data.get("structural_stagnation") or {}).get("stagnated") is True:
            stagnation += 1
        escape = data.get("escape") or {}
        if escape.get("action") == "hybrid":
            hybrid += 1
            trace = escape.get("trace") or {}
            audit = trace.get("two_opt_audit") or {}
            evaluation = trace.get("evaluation") or {}
            validation = evaluation.get("validation") or {}
            if (
                audit.get("selected_edges_exist_in_input_route") is True
                and audit.get("selected_edges_non_adjacent") is True
                and audit.get("exact_single_two_opt_transition") is True
                and validation.get("valid") is True
            ):
                valid_two_opt += 1
        elif escape.get("action") == "restart":
            restart += 1
    return {
        "structural_stagnation_count": stagnation,
        "hybrid_activation_count": hybrid,
        "valid_hybrid_two_opt_count": valid_two_opt,
        "stagnation_restart_count": restart,
    }


def _cost_row(events: Sequence[dict[str, Any]], iteration: int, *, partial: bool) -> dict[str, Any]:
    scoped = _events_for_iteration(events, iteration)
    api_calls, tokens, active, _ = trace_api_metrics(scoped)
    wait = _provider_wait_seconds(scoped)
    total = (float(active) + wait) if active is not None else None
    return {
        "iteration": iteration,
        "label": f"{iteration}*" if partial else str(iteration),
        "partial": partial,
        "api_calls": api_calls,
        "total_tokens": tokens,
        "active_seconds": active,
        "wait_seconds": wait,
        "total_seconds": total,
        "output_failures": sum(event.get("event") == "model_output_attempt" for event in scoped),
        "provider_failures": sum(event.get("event") == "provider_error" for event in scoped),
    }


def build_compact_analysis(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    run_dir = Path(run_dir)
    state = read_state(run_dir / "state.json")
    trace = TraceStore(run_dir / "trace.jsonl")
    shared_root = run_dir.parents[2]
    manifest_path = shared_root / "run.json"
    if not manifest_path.exists():
        raise SystemExit("Compact run için run.json bulunamadı")
    manifest = _read_json(manifest_path)

    raw_config = manifest.get("config") or {}
    experiment_cfg = raw_config.get("experiment") or {}
    problem = manifest.get("problem") or {}
    reference = problem.get("reference_optimum")
    target = experiment_cfg.get("iterations")

    interruption = _first_unresolved_provider_error(trace.events)
    interruption_seq = int(interruption.get("seq") or 0) if interruption else None
    analysis_events = _events_through(trace.events, interruption_seq)
    interruption_iteration = _event_iteration(interruption or {}) if interruption else None

    latest_results: dict[int, dict[str, Any]] = {}
    for event in trace.matching("iteration_result"):
        iteration = int(event.get("iteration") or (event.get("result") or {}).get("iteration") or 0)
        if iteration <= 0:
            continue
        if interruption_seq is not None and int(event.get("seq") or 0) > interruption_seq:
            continue
        if interruption_iteration is not None and iteration >= interruption_iteration:
            continue
        latest_results[iteration] = event.get("result") or {}

    completed_data = [latest_results[key] for key in sorted(latest_results)]
    completed_iterations = len(completed_data)
    completed_set = {int(data["iteration"]) for data in completed_data}

    rows: list[dict[str, Any]] = []
    selection_regrets: list[float] = []
    critic_best_selections = 0

    for data in completed_data:
        iteration = int(data["iteration"])
        observer = data.get("observer_only") or {}
        working_eval = data.get("working_evaluation") or {}
        selected = data.get("selected_before_repair") or {}
        selected_eval = selected.get("evaluation") or {}
        selected_validation = selected_eval.get("validation") or {}
        total_candidates, valid_candidates, critic_best = _candidate_metrics(data)
        selected_distance = _valid_distance(selected_eval)
        selection_regret = (
            selected_distance - critic_best
            if selected_distance is not None and critic_best is not None
            else None
        )
        if selection_regret is not None:
            selection_regrets.append(selection_regret)
            if abs(selection_regret) < 1e-9:
                critic_best_selections += 1

        working_distance = _valid_distance(working_eval)
        observer_best = observer.get("observed_oracle_best_distance")
        working_regret = (
            float(working_distance) - float(observer_best)
            if working_distance is not None and observer_best is not None
            else None
        )
        cost = _cost_row(analysis_events, iteration, partial=False)
        rows.append(
            {
                "iteration": iteration,
                "valid_candidates": valid_candidates,
                "total_candidates": total_candidates,
                "critic_best_distance": critic_best,
                "selected_candidate": selected.get("candidate_id"),
                "selected_valid": selected_validation.get("valid"),
                "working_distance": working_distance,
                "working_gap_percent": _gap(working_distance, reference),
                "observed_oracle_best_so_far": observer_best,
                "working_regret": working_regret,
                "selection_regret": selection_regret,
                "api_calls": cost["api_calls"],
                "total_tokens": cost["total_tokens"],
                "active_seconds": cost["active_seconds"],
                "wait_seconds": cost["wait_seconds"],
                "total_seconds": cost["total_seconds"],
                "output_failures": cost["output_failures"],
                "provider_failures": cost["provider_failures"],
            }
        )

    cost_rows = [_cost_row(analysis_events, row["iteration"], partial=False) for row in rows]
    if interruption_iteration is not None and interruption_iteration not in completed_set:
        partial_cost = _cost_row(analysis_events, interruption_iteration, partial=True)
        if partial_cost["api_calls"] or partial_cost["provider_failures"] or partial_cost["output_failures"]:
            cost_rows.append(partial_cost)

    initializer = _initializer_stats(trace, analysis_events)
    repair = _repair_stats(analysis_events, completed_set)
    recovery = _recovery_stats(analysis_events, completed_data, repair)
    robustness = _robustness_stats(analysis_events)
    adaptive = _adaptive_stats(completed_data)

    total_candidates = sum(int(row["total_candidates"]) for row in rows)
    valid_candidates = sum(int(row["valid_candidates"]) for row in rows)
    valid_possible = sum(int(row["valid_candidates"]) > 0 for row in rows)
    valid_when_possible = sum(
        int(row["valid_candidates"]) > 0 and row.get("selected_valid") is True for row in rows
    )
    avoidable_invalid = sum(
        int(row["valid_candidates"]) > 0 and row.get("selected_valid") is False for row in rows
    )
    comparable = len(selection_regrets)
    selection_count = sum(row.get("selected_candidate") is not None for row in rows)

    state_status = state.get("status")
    is_completed = (
        interruption is None
        and state_status == "completed"
        and (target is None or completed_iterations >= int(target))
    )

    all_api, all_tokens, all_active, all_errors = trace_api_metrics(analysis_events)
    all_wait = _provider_wait_seconds(analysis_events)
    provider_errors = [event for event in analysis_events if event.get("event") == "provider_error"]
    breakdown = Counter()
    for event in provider_errors:
        code = _status_code(event)
        kind = event.get("error_type") or "Error"
        breakdown[f"{code} {kind}" if code is not None else str(kind)] += 1

    final_working = rows[-1]["working_distance"] if rows else initializer.get("accepted_distance")
    final_observer = rows[-1]["observed_oracle_best_so_far"] if rows else initializer.get("accepted_distance")
    final_selected_best = None
    if completed_data:
        final_selected_best = (completed_data[-1].get("observer_only") or {}).get("selected_best_distance")

    summary: dict[str, Any] = {
        "run": {
            "instance": problem.get("name"),
            "dimension": problem.get("dimension"),
            "edge_weight_type": problem.get("edge_weight_type"),
            "provider": state.get("provider"),
            "model": state.get("model"),
            "config": experiment_cfg.get("name"),
            "prompt_set": _prompt_set(raw_config, experiment_cfg),
            "target_iterations": target,
            "completed_iterations": completed_iterations,
            "partial_iteration": interruption_iteration,
        },
        "reference": {"distance": reference},
        "performance": {
            "final_working_distance": final_working,
            # Preserved for compatibility with prior summary consumers.
            "selected_best_distance": final_selected_best,
            "selected_best_gap_percent": _gap(final_selected_best, reference),
            "observed_oracle_best_distance": final_observer,
            "observed_oracle_best_gap_percent": _gap(final_observer, reference),
        },
        "initializer": initializer,
        "critic": {
            "total_candidate_count": total_candidates,
            "valid_candidate_count": valid_candidates,
            "invalid_candidate_count": total_candidates - valid_candidates,
            "valid_iteration_count": valid_possible,
        },
        "scorer": {
            "selection_count": selection_count,
            "valid_possible_iteration_count": valid_possible,
            "valid_selection_when_possible_count": valid_when_possible,
            "avoidable_invalid_selection_count": avoidable_invalid,
            "comparable_iteration_count": comparable,
            "critic_best_selection_count": critic_best_selections,
            "mean_selection_regret": (
                sum(selection_regrets) / comparable if comparable else None
            ),
            "max_selection_regret": max(selection_regrets) if selection_regrets else None,
        },
        "recovery": recovery,
        "model_robustness": robustness,
        "adaptive_search": adaptive,
        "iteration_cost": cost_rows,
        "execution": {
            "api_calls": all_api,
            "total_tokens": all_tokens,
            "active_seconds": all_active,
            "provider_wait_seconds": all_wait,
            "error_count": all_errors,
            "provider_error_breakdown": dict(breakdown),
            "interruption": interruption,
        },
        "termination": {"status": "completed" if is_completed else "partial"},
    }

    report = build_compact_report(summary, rows)
    return summary, rows, report


def _ratio(numerator: int, denominator: int, *, percent: bool = False) -> str:
    if denominator <= 0:
        return "0/0"
    text = f"{numerator}/{denominator}"
    if percent:
        text += f" (%{100.0 * numerator / denominator:.1f})"
    return text


def build_compact_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    run = summary["run"]
    reference = summary["reference"]
    initializer = summary["initializer"]
    critic = summary["critic"]
    scorer = summary["scorer"]
    recovery = summary["recovery"]
    robustness = summary["model_robustness"]
    adaptive = summary["adaptive_search"]
    status = "tamamlandı" if summary["termination"]["status"] == "completed" else "kısmi"

    target = run.get("target_iterations")
    completed = run.get("completed_iterations")
    iteration_text = f"{completed}/{target}" if target is not None else str(completed)

    lines: list[str] = ["AVMA-TSP ANALİZ RAPORU", "=" * 100, ""]
    lines.append(
        _render_table(
            "RUN BİLGİLERİ",
            ["Problem", "Düğüm", "Tür", "Provider", "Model", "Prompt Set", "Referans", "İter.", "Durum"],
            [[
                run.get("instance"), run.get("dimension"), run.get("edge_weight_type"),
                run.get("provider"), run.get("model"), run.get("prompt_set"),
                reference.get("distance"), iteration_text, status,
            ]],
            right_align={1, 6},
            max_widths={4: 26},
        )
    )

    trajectory_rows: list[list[Any]] = []
    for row in rows:
        if row.get("selected_candidate") is None:
            scorer_text = "-"
        else:
            validity = "valid" if row.get("selected_valid") is True else "invalid"
            scorer_text = f"C{row['selected_candidate']} / {validity}"
        trajectory_rows.append([
            row["iteration"],
            f"{row['valid_candidates']}/{row['total_candidates']}",
            row.get("critic_best_distance"),
            scorer_text,
            row.get("working_distance"),
            row.get("working_gap_percent"),
            row.get("observed_oracle_best_so_far"),
            row.get("working_regret"),
        ])

    lines.extend(["", _render_table(
        "SEARCH TRAJECTORY",
        ["İter.", "Critic Valid", "Critic Best", "Scorer", "Working", "Gap %", "Observer Best", "Regret"],
        trajectory_rows,
        right_align={0, 2, 4, 5, 6, 7},
    )])

    initializer_table = _render_table(
        "INITIALIZER",
        ["Metrik", "Sonuç"],
        [
            ["İlk çıktı", "valid" if initializer.get("valid_on_first_attempt") is True else "invalid" if initializer.get("valid_on_first_attempt") is False else "-"],
            ["Repair gerekli", initializer.get("repair_required")],
            ["Repair denemesi", initializer.get("repair_attempt_count")],
            ["Diversity Restart", initializer.get("fallback_restart_count")],
            ["Kabul edilen kaynak", initializer.get("accepted_source")],
            ["Kabul edilen mesafe", initializer.get("accepted_distance")],
        ],
    )
    total_candidates = int(critic.get("total_candidate_count") or 0)
    valid_candidates = int(critic.get("valid_candidate_count") or 0)
    critic_table = _render_table(
        "CRITIC",
        ["Metrik", "Sonuç"],
        [
            ["Toplam aday", total_candidates],
            ["Geçerli", _ratio(valid_candidates, total_candidates, percent=True)],
            ["Geçersiz", _ratio(int(critic.get("invalid_candidate_count") or 0), total_candidates)],
            ["Valid aday bulunan iterasyon", f"{critic.get('valid_iteration_count', 0)}/{completed}"],
        ],
    )
    lines.extend(["", _side_by_side(initializer_table, critic_table)])

    valid_possible = int(scorer.get("valid_possible_iteration_count") or 0)
    valid_selected = int(scorer.get("valid_selection_when_possible_count") or 0)
    comparable = int(scorer.get("comparable_iteration_count") or 0)
    critic_best = int(scorer.get("critic_best_selection_count") or 0)
    scorer_table = _render_table(
        "SCORER",
        ["Metrik", "Sonuç"],
        [
            ["Toplam seçim", scorer.get("selection_count")],
            ["Valid seçimin mümkün olduğu iterasyon", valid_possible],
            ["Bu durumlarda valid seçim", _ratio(valid_selected, valid_possible, percent=True)],
            ["Kaçınılabilir invalid seçim", scorer.get("avoidable_invalid_selection_count")],
            ["Critic-best / comparable", _ratio(critic_best, comparable, percent=True)],
            ["Selection regret ortalama", scorer.get("mean_selection_regret")],
            ["Selection regret maksimum", scorer.get("max_selection_regret")],
        ],
    )
    repair_activations = int(recovery.get("activation_count") or 0)
    recovery_table = _render_table(
        "RECOVERY",
        ["Metrik", "Sonuç"],
        [
            ["Repair aktivasyonu", repair_activations],
            ["Toplam Repair denemesi", recovery.get("total_attempt_count")],
            ["Repair ile kurtarılan", _ratio(int(recovery.get("successful_repair_count") or 0), repair_activations)],
            ["Repair ile kurtarılamayan", _ratio(int(recovery.get("failed_repair_count") or 0), repair_activations)],
            ["Feasibility Restart denemesi", recovery.get("feasibility_restart_attempt_count")],
            ["Restart ile kurtarılan", recovery.get("restart_rescue_count")],
            ["Restart exhaustion", recovery.get("restart_exhaustion_count")],
            ["Incumbent retained", recovery.get("incumbent_retained_count")],
        ],
    )
    lines.extend(["", _side_by_side(scorer_table, recovery_table)])

    by_agent = robustness.get("by_agent") or {}
    robustness_table = _render_table(
        "MODEL ROBUSTNESS",
        ["Olay", "Adet"],
        [
            ["Recoverable model failure", robustness.get("recoverable_model_failure_count")],
            ["├─ Initializer", by_agent.get("initializer", 0)],
            ["├─ Critic", by_agent.get("critic", 0)],
            ["├─ Scorer", by_agent.get("scorer", 0)],
            ["├─ Repair", by_agent.get("repair", 0)],
            ["├─ Diversity", by_agent.get("diversity", 0)],
            ["└─ Hybrid", by_agent.get("hybrid", 0)],
            ["", ""],
            ["Parse / malformed output", robustness.get("parse_failure_count")],
            ["Timeout", robustness.get("timeout_count")],
            ["Transient HTTP", robustness.get("transient_http_count")],
        ],
        right_align={1},
    )
    hybrid_count = int(adaptive.get("hybrid_activation_count") or 0)
    adaptive_table = _render_table(
        "ADAPTIVE SEARCH",
        ["Metrik", "Sonuç"],
        [
            ["Structural stagnation", adaptive.get("structural_stagnation_count")],
            ["Hybrid activation", hybrid_count],
            ["Valid exact 2-opt", _ratio(int(adaptive.get("valid_hybrid_two_opt_count") or 0), hybrid_count)],
            ["Stagnation Diversity Restart", adaptive.get("stagnation_restart_count")],
        ],
    )
    lines.extend(["", _side_by_side(robustness_table, adaptive_table)])

    cost_rows = [
        [
            item.get("label"), item.get("api_calls"), item.get("total_tokens"),
            item.get("active_seconds"), item.get("wait_seconds"), item.get("total_seconds"),
            item.get("output_failures"), item.get("provider_failures"),
        ]
        for item in summary.get("iteration_cost") or []
    ]
    lines.extend(["", _render_table(
        "ITERATION COST",
        ["İter.", "API", "Token", "Aktif sn", "Wait sn", "Total sn", "Output Fail", "Provider Fail"],
        cost_rows,
        right_align={0, 1, 2, 3, 4, 5, 6, 7},
    )])

    lines.extend(["", "=" * 100])
    return "\n".join(lines) + "\n"


def write_compact_analysis(
    run_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    report: str,
) -> dict[str, Path]:
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for legacy_name in ("analysis_summary.json", "observed_oracle_best.png"):
        legacy_path = analysis_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = analysis_dir / "iterations.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report_path = analysis_dir / "analysis_report.txt"
    report_path.write_text(report, encoding="utf-8")

    plot_path = analysis_dir / "selected_vs_oracle.png"
    x = [row["iteration"] for row in rows]
    working = [row["working_distance"] for row in rows]
    oracle = [row["observed_oracle_best_so_far"] for row in rows]
    reference = summary["reference"].get("distance")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if x:
        ax.plot(x, working, marker="o", markerfacecolor="none", markersize=6, linewidth=1.8, label="Working")
        ax.plot(x, oracle, linestyle="--", marker="x", markersize=6, linewidth=1.6, label="Observed oracle best-so-far")
        ax.set_xticks(x)
    if reference is not None:
        ax.axhline(float(reference), linestyle=":", label="Reference optimum")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance")
    ax.set_title(f"{summary['run'].get('instance')} — Working vs Observed Oracle")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {"summary": summary_path, "iterations": csv_path, "plot": plot_path, "report": report_path}
