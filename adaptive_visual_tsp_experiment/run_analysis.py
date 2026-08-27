from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiment.compact import TraceStore, read_state, trace_api_metrics
from src.experiment.layout import (
    discover_model_runs,
    is_compact_model_run,
    is_legacy_run,
    model_run_labels,
    provider_model_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one AVMA-TSP provider/model run")
    parser.add_argument("--run-dir", required=True, help="Legacy run, model directory, or shared run root")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def resolve_analysis_run_dir(
    run_dir: str | Path,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    path = Path(run_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if is_legacy_run(path) or is_compact_model_run(path):
        return path

    if (provider is None) != (model is None):
        raise SystemExit("Shared run analizi için --provider ve --model birlikte verilmelidir")

    if provider is not None and model is not None:
        selected = provider_model_dir(path, provider, model)
        if not (is_compact_model_run(selected) or (selected / "run_manifest.json").exists()):
            raise SystemExit(f"Provider/model run bulunamadı: {provider}/{model}")
        return selected

    discovered = discover_model_runs(path)
    if len(discovered) == 1:
        return discovered[0]
    if not discovered:
        raise SystemExit("Analiz edilebilir provider/model sonucu bulunamadı")
    labels = ", ".join(model_run_labels(discovered))
    raise SystemExit(
        "Bu shared run birden fazla model içeriyor. "
        f"--provider ve --model seçin. Mevcut: {labels}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _valid_distance(evaluation: dict[str, Any] | None) -> float | None:
    if not isinstance(evaluation, dict):
        return None
    validation = evaluation.get("validation") or {}
    distance = evaluation.get("distance")
    if validation.get("valid") is True and distance is not None:
        return float(distance)
    return None


def _candidate_metrics(data: dict[str, Any]) -> tuple[int, int, float | None]:
    candidates = data.get("critic_candidates") or []
    valid_distances: list[float] = []
    valid_count = 0
    for candidate in candidates:
        distance = _valid_distance(candidate.get("evaluation"))
        if distance is not None:
            valid_count += 1
            valid_distances.append(distance)
    return len(candidates), valid_count, min(valid_distances) if valid_distances else None


def _selection_regret(selected_distance: float | None, iteration_oracle: float | None) -> float | None:
    if selected_distance is None or iteration_oracle is None:
        return None
    return selected_distance - iteration_oracle


def _legacy_execution_metrics(root: Path) -> tuple[int, int | None, float | None, int]:
    api_calls = 0
    total_tokens = 0
    saw_tokens = False
    active_seconds = 0.0
    latency_complete = True

    for call_path in sorted(root.rglob("*_call.json")):
        call = _read_json(call_path)
        metadata = call.get("raw_metadata") or {}
        native_count = metadata.get("native_candidate_count")
        candidate_index = metadata.get("candidate_index")
        native_secondary = (
            isinstance(native_count, int)
            and native_count > 1
            and isinstance(candidate_index, int)
            and candidate_index > 1
        )
        if native_secondary:
            continue
        api_calls += 1
        usage = call.get("usage") or {}
        token_count = usage.get("total_token_count")
        if token_count is not None:
            total_tokens += int(token_count)
            saw_tokens = True
        latency = call.get("latency_seconds")
        if latency is None:
            latency_complete = False
        else:
            active_seconds += float(latency)

    for response_path in sorted(root.rglob("output_attempts/attempt_*/provider_response.json")):
        response = _read_json(response_path)
        api_calls += 1
        usage = response.get("usage") or {}
        token_count = usage.get("total_token_count")
        if token_count is not None:
            total_tokens += int(token_count)
            saw_tokens = True
        latency = response.get("latency_seconds")
        if latency is None:
            latency_complete = False
        else:
            active_seconds += float(latency)

    errors = sum(1 for _ in root.rglob("provider_error.json"))
    errors += sum(1 for _ in root.rglob("output_attempts/attempt_*/parse_error.json"))
    tokens = total_tokens if saw_tokens else None
    active = active_seconds if api_calls > 0 and latency_complete else None
    return api_calls, tokens, active, errors


def _legacy_provider_errors(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("provider_error.json")):
        try:
            item = _read_json(path)
        except Exception:
            continue
        item["artifact"] = str(path.relative_to(root))
        result.append(item)
    return result


def _load_artifacts(run_dir: Path) -> dict[str, Any]:
    if is_compact_model_run(run_dir):
        state = read_state(run_dir / "state.json")
        trace = TraceStore(run_dir / "trace.jsonl")
        shared_root = run_dir.parents[2]
        manifest_path = shared_root / "run.json"
        if not manifest_path.exists():
            raise SystemExit("Compact run için run.json bulunamadı")
        manifest = _read_json(manifest_path)
        iteration_events = trace.matching("iteration_result")
        iteration_data = [event.get("result") or {} for event in iteration_events]
        initializer_event = trace.find_last("initializer_result") or {}
        initializer = initializer_event.get("result") or {}
        provider = {"name": state.get("provider"), "model": state.get("model")}
        return {
            "mode": "compact",
            "manifest": manifest,
            "provider": provider,
            "state": state,
            "trace": trace,
            "iteration_data": iteration_data,
            "initializer": initializer,
            "initializer_event": initializer_event,
            "run_summary": state.get("summary") or {},
        }

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("run_manifest.json bulunamadı")
    manifest = _read_json(manifest_path)
    iteration_files = sorted((run_dir / "iterations").glob("iteration_*/iteration_result.json"))
    if not iteration_files:
        raise SystemExit("iteration_result.json bulunamadı")
    initializer_path = run_dir / "initializer" / "initializer_result.json"
    return {
        "mode": "legacy",
        "manifest": manifest,
        "provider": manifest.get("provider") or {},
        "state": {},
        "trace": None,
        "iteration_files": iteration_files,
        "iteration_data": [_read_json(path) for path in iteration_files],
        "initializer": _read_json(initializer_path) if initializer_path.exists() else {},
        "initializer_event": {},
        "run_summary": _read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {},
    }


def _initializer_stats(run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    initializer = artifacts["initializer"]
    mode = artifacts["mode"]
    initial_eval = initializer.get("evaluation") or {}
    initial_valid = (initial_eval.get("validation") or {}).get("valid")
    initial_distance = initial_eval.get("distance")

    if mode == "compact":
        trace: TraceStore = artifacts["trace"]
        initial_event = trace.find_last("initializer_candidate") or {}
        if initial_event:
            initial_eval = initial_event.get("evaluation") or {}
            initial_valid = (initial_eval.get("validation") or {}).get("valid")
            initial_distance = initial_eval.get("distance")
        direct_repairs = [
            event for event in trace.matching("repair_result")
            if event.get("scope") == "initializer"
        ]
        fallback = [
            event for event in trace.matching("diversity_result")
            if event.get("scope") == "initializer.fallback"
        ]
        accepted_eval = (artifacts.get("initializer_event") or {}).get("evaluation") or {}
    else:
        repair_dir = run_dir / "initializer" / "repair"
        direct_repairs = sorted(repair_dir.glob("attempt_*/repair_result.json")) if repair_dir.exists() else []
        fallback_dir = run_dir / "initializer" / "fallback_restart"
        fallback = sorted(fallback_dir.glob("restart_*/diversity_result.json")) if fallback_dir.exists() else []
        accepted_eval = {}
        if initial_valid is True:
            accepted_eval = initial_eval
        elif isinstance(initializer.get("repair"), list) and initializer["repair"]:
            accepted_eval = initializer["repair"][-1].get("evaluation") or {}
        else:
            restart = initializer.get("restart") or {}
            attempts = restart.get("attempts") or []
            if attempts:
                last = attempts[-1]
                accepted_eval = last.get("evaluation") or {}
                repair = last.get("repair")
                if isinstance(repair, list) and repair:
                    accepted_eval = repair[-1].get("evaluation") or accepted_eval

    repair_attempts = len(direct_repairs)
    if repair_attempts == 0:
        repair_result = "yok"
    else:
        if mode == "compact":
            last_eval = direct_repairs[-1].get("evaluation") or {}
        else:
            last_eval = _read_json(direct_repairs[-1]).get("evaluation") or {}
        repair_result = "başarılı" if (last_eval.get("validation") or {}).get("valid") is True else "başarısız"

    fallback_count = len(fallback)
    if initial_valid is True:
        accepted_source = "initializer"
    elif repair_result == "başarılı":
        accepted_source = "repair"
    elif fallback_count:
        accepted_source = "fallback restart"
    else:
        accepted_source = "-"

    return {
        "valid_on_first_attempt": initial_valid,
        "repair_required": initial_valid is False if initial_valid is not None else None,
        "repair_attempt_count": repair_attempts,
        "repair_result": repair_result,
        "fallback_restart_count": fallback_count,
        "accepted_source": accepted_source,
        "initial_distance": initial_distance,
        "accepted_distance": accepted_eval.get("distance"),
    }


def _iteration_from_scope(scope: str) -> int | None:
    import re

    match = re.search(r"iteration_(\d+)", scope)
    return int(match.group(1)) if match else None


def _repair_stats(run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    """Summarize only repair chains that happen inside search iterations.

    Initializer repair/fallback belongs exclusively to the INITIALIZER section
    and is deliberately excluded here to avoid double-reporting.
    """

    groups: dict[str, list[dict[str, Any]]] = {}

    if artifacts["mode"] == "compact":
        trace: TraceStore = artifacts["trace"]
        for event in trace.matching("repair_result"):
            scope = str(event.get("scope") or "")
            if not scope.startswith("iteration_"):
                continue
            groups.setdefault(scope, []).append(event)
    else:
        iterations_root = run_dir / "iterations"
        if iterations_root.exists():
            for path in sorted(iterations_root.rglob("attempt_*/repair_result.json")):
                rel = path.relative_to(run_dir).as_posix()
                # One repair activation is one parent repair chain. The path may
                # be nested under selected repair, restart or hybrid escape.
                scope = str(path.parent.parent.relative_to(run_dir))
                data = _read_json(path)
                groups.setdefault(scope, []).append(
                    {
                        "scope": scope,
                        "iteration": _iteration_from_artifact(rel),
                        "evaluation": data.get("evaluation") or {},
                    }
                )

    successes = 0
    failures = 0
    details: dict[int, dict[str, int]] = {}

    for scope, group in groups.items():
        last_eval = group[-1].get("evaluation") or {}
        valid = (last_eval.get("validation") or {}).get("valid") is True
        if valid:
            successes += 1
        else:
            failures += 1

        iteration = next(
            (int(event["iteration"]) for event in group if event.get("iteration") is not None),
            _iteration_from_scope(scope),
        )
        if iteration is not None:
            item = details.setdefault(
                iteration,
                {"activation_count": 0, "attempt_count": 0, "successful_count": 0, "failed_count": 0},
            )
            item["activation_count"] += 1
            item["attempt_count"] += len(group)
            item["successful_count" if valid else "failed_count"] += 1

    return {
        "activation_count": len(groups),
        "successful_repair_count": successes,
        "failed_repair_count": failures,
        "total_attempt_count": sum(len(group) for group in groups.values()),
        "iterations": [
            {"iteration": iteration, **values}
            for iteration, values in sorted(details.items())
        ],
    }


def _hybrid_two_opt_stats(iteration_data: list[dict[str, Any]]) -> tuple[int, int]:
    activations = 0
    valid_two_opt = 0
    for data in iteration_data:
        escape = data.get("escape") or {}
        if escape.get("action") != "hybrid":
            continue
        activations += 1
        trace = escape.get("trace") or {}
        audit = trace.get("two_opt_audit") or {}
        evaluation = trace.get("evaluation") or {}
        validation = evaluation.get("validation") or {}
        audit_ok = (
            audit.get("selected_edges_exist_in_input_route") is True
            and audit.get("selected_edges_non_adjacent") is True
            and audit.get("exact_single_two_opt_transition") is True
        )
        if audit_ok and validation.get("valid") is True:
            valid_two_opt += 1
    return activations, valid_two_opt


def _execution_summary(run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    if artifacts["mode"] == "compact":
        trace: TraceStore = artifacts["trace"]
        api_calls, tokens, active, errors = trace_api_metrics(trace.events)
        provider_errors = trace.matching("provider_error")
        state = artifacts["state"]
        current = state.get("current") or {}
    else:
        api_calls, tokens, active, errors = _legacy_execution_metrics(run_dir)
        provider_errors = _legacy_provider_errors(run_dir)
        current = {}
        if provider_errors:
            last = provider_errors[-1]
            current = {
                "phase": last.get("phase"),
                "iteration": _iteration_from_artifact(last.get("artifact")),
                "candidate": _candidate_from_artifact(last.get("artifact")),
            }

    breakdown = Counter()
    for error in provider_errors:
        code = error.get("status_code")
        kind = error.get("error_type") or "Error"
        key = f"{code} {kind}" if code is not None else str(kind)
        breakdown[key] += 1

    return {
        "api_calls": api_calls,
        "total_tokens": tokens,
        "active_seconds": active,
        "error_count": errors,
        "provider_error_breakdown": dict(breakdown),
        "last_stage": current,
    }


def _iteration_from_artifact(value: Any) -> int | None:
    import re
    match = re.search(r"iteration_(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def _candidate_from_artifact(value: Any) -> int | None:
    import re
    match = re.search(r"candidate_(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def build_analysis(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    artifacts = _load_artifacts(run_dir)
    manifest = artifacts["manifest"]
    iteration_data = artifacts["iteration_data"]
    if not iteration_data:
        raise SystemExit("Tamamlanmış iteration_result kaydı bulunamadı")

    problem = manifest.get("problem") or {}
    provider = artifacts["provider"]
    raw_config = manifest.get("config") or {}
    experiment_cfg = raw_config.get("experiment") or {}
    reference = problem.get("reference_optimum")

    def gap(distance: Any) -> float | None:
        if distance is None or reference is None or float(reference) <= 0:
            return None
        return ((float(distance) - float(reference)) / float(reference)) * 100.0

    rows: list[dict[str, Any]] = []
    scorer_oracle_selection_count = 0
    regret_values: list[float] = []

    for data in iteration_data:
        iteration = int(data["iteration"])
        observer = data.get("observer_only") or {}
        working = data.get("working_evaluation") or {}
        working_validation = working.get("validation") or {}
        selected = data.get("selected_before_repair") or {}
        selected_eval = selected.get("evaluation") or {}
        selected_validation = selected_eval.get("validation") or {}
        stagnation = data.get("structural_stagnation") or {}
        escape = data.get("escape") or {}

        total_candidates, valid_candidates, iteration_oracle = _candidate_metrics(data)
        selected_distance = _valid_distance(selected_eval)
        regret = _selection_regret(selected_distance, iteration_oracle)
        if regret is not None:
            regret_values.append(regret)
            if abs(regret) < 1e-9:
                scorer_oracle_selection_count += 1

        if artifacts["mode"] == "compact":
            trace: TraceStore = artifacts["trace"]
            iteration_events = [event for event in trace.events if event.get("iteration") == iteration]
            api_calls, total_tokens, active_seconds, provider_errors = trace_api_metrics(iteration_events)
        else:
            iteration_dir = run_dir / "iterations" / f"iteration_{iteration:03d}"
            api_calls, total_tokens, active_seconds, provider_errors = _legacy_execution_metrics(iteration_dir)

        escape_action = escape.get("action")
        rows.append(
            {
                "iteration": iteration,
                "selected_candidate": selected.get("candidate_id"),
                "selected_valid": selected_validation.get("valid"),
                "selected_distance": selected_distance,
                "selected_gap_percent": selected_eval.get("gap_percent") if selected_distance is not None else None,
                "working_distance": working.get("distance"),
                "working_gap_percent": working.get("gap_percent"),
                "iteration_oracle_distance": iteration_oracle,
                "selection_regret": regret,
                "selected_best_so_far": observer.get("selected_best_distance"),
                "system_gbest_gap_percent": gap(observer.get("selected_best_distance")),
                "observed_oracle_best_so_far": observer.get("observed_oracle_best_distance"),
                "working_crossings": working.get("crossings"),
                "valid_candidates": valid_candidates,
                "total_candidates": total_candidates,
                "repair_used": selected_validation.get("valid") is False,
                "stagnation_detected": bool(stagnation.get("stagnated")),
                "mean_similarity": stagnation.get("mean_consecutive_similarity"),
                "hybrid_used": escape_action == "hybrid",
                "restart_used": escape_action == "restart",
                "escape_action": escape_action,
                "working_valid": working_validation.get("valid"),
                "api_calls": api_calls,
                "total_tokens": total_tokens,
                "active_seconds": active_seconds,
                "provider_errors": provider_errors,
            }
        )

    final_selected = rows[-1]["selected_best_so_far"]
    final_oracle = rows[-1]["observed_oracle_best_so_far"]
    initializer_stats = _initializer_stats(run_dir, artifacts)
    repair_stats = _repair_stats(run_dir, artifacts)
    invalid_selected = sum(row["selected_valid"] is False for row in rows)
    stagnation_count = sum(row["stagnation_detected"] for row in rows)
    hybrid_count, valid_hybrid_two_opt_count = _hybrid_two_opt_stats(iteration_data)
    search_restart_count = sum(row["restart_used"] for row in rows)
    total_candidates = sum(int(row["total_candidates"]) for row in rows)
    valid_candidates = sum(int(row["valid_candidates"]) for row in rows)
    critic_best_distances = [
        row["iteration_oracle_distance"]
        for row in rows
        if row.get("iteration_oracle_distance") is not None
    ]

    configured_iterations = experiment_cfg.get("iterations")
    completed_iterations = len(rows)
    state_status = (artifacts.get("state") or {}).get("status")
    completed = state_status == "completed" or (
        bool(artifacts["run_summary"])
        and (configured_iterations is None or completed_iterations >= int(configured_iterations))
    )
    execution = _execution_summary(run_dir, artifacts)

    summary: dict[str, Any] = {
        "run": {
            "instance": problem.get("name"),
            "dimension": problem.get("dimension"),
            "edge_weight_type": problem.get("edge_weight_type"),
            "provider": provider.get("name"),
            "model": provider.get("model"),
            "config": experiment_cfg.get("name"),
            "target_iterations": configured_iterations,
            "completed_iterations": completed_iterations,
        },
        "reference": {"distance": reference},
        "performance": {
            "final_working_distance": rows[-1]["working_distance"],
            "selected_best_distance": final_selected,
            "selected_best_gap_percent": gap(final_selected),
            "observed_oracle_best_distance": final_oracle,
            "observed_oracle_best_gap_percent": gap(final_oracle),
        },
        "initializer": initializer_stats,
        "critic": {
            "total_candidate_count": total_candidates,
            "valid_candidate_count": valid_candidates,
            "invalid_candidate_count": total_candidates - valid_candidates,
            "iteration_best_distances": critic_best_distances,
        },
        "scorer": {
            "selection_count": completed_iterations,
            "invalid_selection_count": invalid_selected,
            "oracle_selection_count": scorer_oracle_selection_count,
            "mean_selection_regret": sum(regret_values) / len(regret_values) if regret_values else None,
            "max_selection_regret": max(regret_values) if regret_values else None,
        },
        "repair": repair_stats,
        "adaptive_search": {
            "structural_stagnation_count": stagnation_count,
            "hybrid_activation_count": hybrid_count,
            "valid_hybrid_two_opt_count": valid_hybrid_two_opt_count,
            "diversity_restart_count": search_restart_count,
            "initializer_fallback_restart_count": initializer_stats["fallback_restart_count"],
        },
        "execution": execution,
        "termination": {"status": "completed" if completed else "partial"},
    }

    report = build_report(summary, rows)
    return summary, rows, report


def _last_stage_text(stage: dict[str, Any]) -> str:
    if not stage:
        return "-"
    parts: list[str] = []
    if stage.get("iteration") is not None:
        parts.append(f"Iteration {stage['iteration']}")
    if stage.get("phase"):
        parts.append(str(stage["phase"]))
    if stage.get("candidate") is not None:
        parts.append(f"C{stage['candidate']}")
    if stage.get("scope") and not parts:
        parts.append(str(stage["scope"]))
    return " / ".join(parts) if parts else "-"


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    run = summary["run"]
    reference = summary["reference"]
    initializer = summary["initializer"]
    critic = summary["critic"]
    scorer = summary["scorer"]
    repair = summary["repair"]
    adaptive = summary["adaptive_search"]
    execution = summary["execution"]
    termination = summary["termination"]

    status = {"completed": "tamamlandı", "partial": "kısmi"}.get(
        str(termination.get("status")), termination.get("status")
    )

    lines: list[str] = ["AVMA-TSP ANALİZ RAPORU", "=" * 80, ""]
    lines.append(
        _render_table(
            "RUN BİLGİLERİ",
            ["Problem", "Düğüm", "Tür", "Provider", "Model", "Referans", "İter.", "Durum"],
            [[
                run.get("instance"), run.get("dimension"), run.get("edge_weight_type"),
                run.get("provider"), run.get("model"), reference.get("distance"),
                run.get("completed_iterations"), status,
            ]],
            right_align={1, 5, 6},
            max_widths={4: 26},
        )
    )

    lines.extend(["", "INITIALIZER", ""])
    lines.append(f"  İlk çıktı geçerli       = {_cell(initializer.get('valid_on_first_attempt'))}")
    lines.append(f"  Repair gerekli          = {_cell(initializer.get('repair_required'))}")
    lines.append(f"  Repair denemesi         = {_cell(initializer.get('repair_attempt_count'))}")
    lines.append(f"  Repair sonucu           = {_cell(initializer.get('repair_result'))}")
    lines.append(f"  Fallback restart        = {_cell(initializer.get('fallback_restart_count'))}")
    lines.append(f"  Kabul edilen kaynak     = {_cell(initializer.get('accepted_source'))}")
    lines.append(f"  Başlangıç mesafesi      = {_cell(initializer.get('accepted_distance'))}")

    iteration_rows: list[list[Any]] = []
    for row in rows:
        scorer_choice = f"C{row['selected_candidate']}" if row.get("selected_candidate") is not None else "-"
        iteration_rows.append(
            [
                row["iteration"],
                f"{row['valid_candidates']}/{row['total_candidates']}",
                scorer_choice,
                row.get("selected_distance"),
                row.get("selected_best_so_far"),
                row.get("observed_oracle_best_so_far"),
                row.get("system_gbest_gap_percent"),
                row.get("selection_regret"),
                row.get("api_calls"),
                row.get("total_tokens"),
                row.get("active_seconds"),
                row.get("provider_errors"),
            ]
        )

    lines.extend(["", "İTERASYONLAR", ""])
    lines.append(
        _render_table(
            "",
            [
                "İter.", "Critic Valid", "Scorer", "Seçilen", "Sistem GBest",
                "Oracle GBest", "Gap %", "Regret", "API", "Token", "Aktif sn", "Hata",
            ],
            iteration_rows,
            right_align={0, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        ).lstrip("\n")
    )

    lines.extend(["", "REPAIR", ""])
    lines.append(f"  İterasyon içi aktivasyon = {_cell(repair.get('activation_count'))}")
    lines.append(f"  Toplam deneme             = {_cell(repair.get('total_attempt_count'))}")
    lines.append(f"  Başarılı                  = {_cell(repair.get('successful_repair_count'))}")
    lines.append(f"  Başarısız                 = {_cell(repair.get('failed_repair_count'))}")
    for item in repair.get("iterations") or []:
        iteration = item.get("iteration")
        activations = item.get("activation_count")
        attempts = item.get("attempt_count")
        success = item.get("successful_count")
        failure = item.get("failed_count")
        lines.append(
            f"  İterasyon {iteration:<3}             = "
            f"{activations} aktivasyon / {attempts} deneme / "
            f"{success} başarılı / {failure} başarısız"
        )

    total_candidates = int(critic.get("total_candidate_count") or 0)
    valid_candidates = int(critic.get("valid_candidate_count") or 0)
    valid_percent = (
        f"{valid_candidates}/{total_candidates} (%{100.0 * valid_candidates / total_candidates:.1f})"
        if total_candidates else "0/0"
    )
    critic_bests = critic.get("iteration_best_distances") or []
    critic_best_text = " → ".join(_cell(value) for value in critic_bests) if critic_bests else "-"

    lines.extend(["", "CRITIC", ""])
    lines.append(f"  Toplam aday           = {total_candidates}")
    lines.append(f"  Geçerli               = {valid_percent}")
    lines.append(f"  Geçersiz              = {critic.get('invalid_candidate_count', 0)}")
    lines.append(f"  İterasyon bestleri    = {critic_best_text}")

    scorer_count = int(scorer.get("selection_count") or 0)
    oracle_count = int(scorer.get("oracle_selection_count") or 0)
    oracle_selection_text = (
        f"{oracle_count}/{scorer_count} (%{100.0 * oracle_count / scorer_count:.1f})"
        if scorer_count else "0/0"
    )

    lines.extend(["", "SCORER", ""])
    lines.append(f"  Toplam seçim          = {scorer_count}")
    lines.append(f"  Geçersiz seçim        = {_cell(scorer.get('invalid_selection_count'))}")
    lines.append(f"  Critic Best seçimi    = {oracle_selection_text}")
    lines.append(f"  Ortalama regret       = {_cell(scorer.get('mean_selection_regret'))}")
    lines.append(f"  Maksimum regret       = {_cell(scorer.get('max_selection_regret'))}")

    hybrid_count = int(adaptive.get("hybrid_activation_count") or 0)
    valid_hybrid = int(adaptive.get("valid_hybrid_two_opt_count") or 0)
    lines.extend(["", "HYBRID & RESTART", ""])
    lines.append(f"  Structural stagnation = {_cell(adaptive.get('structural_stagnation_count'))}")
    lines.append(f"  Hybrid                = {hybrid_count}")
    lines.append(f"  Geçerli Hybrid 2-opt  = {valid_hybrid}/{hybrid_count}")
    lines.append(f"  Search restart        = {_cell(adaptive.get('diversity_restart_count'))}")

    lines.extend(["", "RUN EXECUTION", ""])
    target = run.get("target_iterations")
    completed = run.get("completed_iterations")
    target_text = f"{completed}/{target}" if target is not None else str(completed)
    lines.append(f"  Tamamlanan iterasyon    = {target_text}")
    lines.append(f"  Toplam API çağrısı      = {_cell(execution.get('api_calls'))}")
    lines.append(f"  Toplam token            = {_cell(execution.get('total_tokens'))}")
    lines.append(f"  Toplam aktif süre (sn)  = {_cell(execution.get('active_seconds'))}")
    lines.append(f"  Kayıtlı hata            = {_cell(execution.get('error_count'))}")
    breakdown = execution.get("provider_error_breakdown") or {}
    if breakdown:
        lines.append("  Provider hataları")
        for name, count in breakdown.items():
            lines.append(f"    {name:<22} = {count}")
    lines.append(f"  Son ulaşılan aşama      = {_last_stage_text(execution.get('last_stage') or {})}")
    lines.append(f"  Durum                    = {status}")

    lines.extend(["", "=" * 80])
    return "\n".join(lines) + "\n"


def write_analysis(run_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]], report: str) -> dict[str, Path]:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for legacy_name in ("analysis_summary.json", "observed_oracle_best.png"):
        legacy_path = analysis_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = analysis_dir / "iterations.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report_path = analysis_dir / "analysis_report.txt"
    report_path.write_text(report, encoding="utf-8")

    plot_path = analysis_dir / "selected_vs_oracle.png"
    x = [row["iteration"] for row in rows]
    selected = [row["selected_best_so_far"] for row in rows]
    oracle = [row["observed_oracle_best_so_far"] for row in rows]
    reference = summary["reference"].get("distance")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        x,
        selected,
        marker="o",
        markerfacecolor="none",
        markersize=6,
        linewidth=1.8,
        label="Selected best-so-far",
    )
    ax.plot(
        x,
        oracle,
        linestyle="--",
        marker="x",
        markersize=6,
        linewidth=1.6,
        label="Observed oracle best-so-far",
    )
    if reference is not None:
        ax.axhline(float(reference), linestyle=":", label="Reference optimum")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance")
    ax.set_title(f"{summary['run'].get('instance')} — Selected vs Observed Oracle")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {"summary": summary_path, "iterations": csv_path, "plot": plot_path, "report": report_path}


def main() -> None:
    args = parse_args()
    run_dir = resolve_analysis_run_dir(args.run_dir, provider=args.provider, model=args.model)
    summary, rows, report = build_analysis(run_dir)
    outputs = write_analysis(run_dir, summary, rows, report)

    print(report, end="")
    print("\nDosyalar")
    print(f"  summary.json          : {outputs['summary']}")
    print(f"  iterations.csv        : {outputs['iterations']}")
    print(f"  selected_vs_oracle.png: {outputs['plot']}")
    print(f"  analysis_report.txt   : {outputs['report']}")


if __name__ == "__main__":
    main()
