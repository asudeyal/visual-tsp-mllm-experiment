from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _find_project() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd / "avma_cvrp_experiment"):
        if (candidate / "run_analysis.py").exists() and (candidate / "src").exists():
            return candidate
    raise SystemExit(
        "avma_cvrp_experiment bulunamadı. "
        "Komutu repo kökünden veya avma_cvrp_experiment klasöründen çalıştır."
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for lineno, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path} satır {lineno} JSON değil: {exc}") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _truncate(text: str, limit: int) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _table(
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    *,
    max_widths: dict[int, int] | None = None,
) -> str:
    limits = max_widths or {}
    rendered: list[list[str]] = []
    for row in rows:
        values = [_fmt(value) for value in row]
        for i, value in enumerate(values):
            if i in limits:
                values[i] = _truncate(value, limits[i])
        rendered.append(values)

    widths = [len(h) for h in headers]
    for row in rendered:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def border(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def line(values: list[str]) -> str:
        return "│" + "│".join(f" {values[i].ljust(widths[i])} " for i in range(len(headers))) + "│"

    out = [title, border("┌", "┬", "┐"), line(headers), border("├", "┼", "┤")]
    if rendered:
        out.extend(line(row) for row in rendered)
    else:
        empty = ["(no records)"] + [""] * (len(headers) - 1)
        out.append(line(empty))
    out.append(border("└", "┴", "┘"))
    return "\n".join(out)


def _event_iteration(event: dict[str, Any]) -> int | None:
    value = event.get("iteration")
    if isinstance(value, int):
        return value
    scope = str(event.get("scope") or "")
    marker = "iteration_"
    pos = scope.find(marker)
    if pos < 0:
        return None
    digits = ""
    for ch in scope[pos + len(marker):]:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def _eval_metrics(evaluation: dict[str, Any] | None) -> dict[str, Any]:
    evaluation = evaluation or {}
    validation = evaluation.get("validation") or {}
    ratios_raw = validation.get("route_capacity_ratios") or []
    ratios: list[float] = []
    for value in ratios_raw:
        try:
            ratios.append(float(value))
        except (TypeError, ValueError):
            pass
    excesses = [max(0.0, ratio - 1.0) for ratio in ratios]
    exceeded = validation.get("capacity_exceeded_route_indices") or []
    return {
        "valid": validation.get("valid"),
        "reasons": validation.get("reasons") or [],
        "route_loads": validation.get("route_loads") or [],
        "ratios": ratios_raw,
        "violations": len(exceeded),
        "excess_sum": sum(excesses) if ratios else None,
        "excess_max": max(excesses) if ratios else None,
        "missing": validation.get("missing_nodes") or [],
        "duplicates": validation.get("duplicate_nodes") or [],
        "unknown": validation.get("unknown_nodes") or [],
        "vehicle_count": validation.get("vehicle_count"),
        "distance": evaluation.get("distance"),
        "gap": evaluation.get("gap_percent"),
        "crossings": evaluation.get("crossings"),
    }


def _call_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "agent_call":
            continue
        call = event.get("call") or {}
        usage = call.get("usage") or {}
        rows.append({
            "seq": event.get("seq"),
            "agent": event.get("agent"),
            "phase": call.get("phase"),
            "scope": event.get("scope"),
            "iteration": _event_iteration(event),
            "candidate": event.get("candidate"),
            "attempt": event.get("attempt"),
            "restart_attempt": event.get("restart_attempt"),
            "latency": call.get("latency_seconds"),
            "prompt_tokens": usage.get("prompt_token_count"),
            "candidate_tokens": usage.get("candidates_token_count"),
            "thought_tokens": usage.get("thoughts_token_count"),
            "total_tokens": usage.get("total_token_count"),
        })
    return rows


def _initializer_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for event in events:
        kind = event.get("event")
        scope = str(event.get("scope") or "")
        stage = None
        route = None
        evaluation = None

        if kind == "initializer_candidate":
            stage = "Initializer"
            route = event.get("route")
            evaluation = event.get("evaluation")

        elif kind == "repair_result" and scope == "initializer":
            stage = f"Initializer Repair {event.get('attempt') or '?'}"
            route = event.get("output_route")
            evaluation = event.get("evaluation")

        elif kind == "diversity_result" and scope == "initializer.fallback":
            stage = f"Fallback Restart {event.get('restart_attempt') or '?'}"
            route = event.get("route")
            evaluation = event.get("evaluation")

        elif kind == "repair_result" and scope.startswith("initializer.fallback.restart_"):
            restart = "?"
            tail = scope.split("initializer.fallback.restart_", 1)[1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if digits:
                restart = str(int(digits))
            stage = f"Restart {restart} Repair {event.get('attempt') or '?'}"
            route = event.get("output_route")
            evaluation = event.get("evaluation")

        elif kind == "initializer_result":
            stage = "ACCEPTED"
            route = event.get("accepted_route")
            evaluation = event.get("evaluation")

        if stage is None:
            continue

        metrics = _eval_metrics(evaluation)
        rows.append({
            "seq": event.get("seq"),
            "stage": stage,
            "valid": metrics["valid"],
            "violations": metrics["violations"],
            "excess_sum": metrics["excess_sum"],
            "excess_max": metrics["excess_max"],
            "loads": metrics["route_loads"],
            "ratios": metrics["ratios"],
            "reasons": metrics["reasons"],
            "missing": metrics["missing"],
            "duplicates": metrics["duplicates"],
            "unknown": metrics["unknown"],
            "crossings": metrics["crossings"],
            "distance": metrics["distance"],
            "route": route,
        })

    return rows


def _critic_by_iteration(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("event") != "critic_result":
            continue
        iteration = _event_iteration(event)
        if iteration is None:
            continue
        metrics = _eval_metrics(event.get("evaluation"))
        grouped[iteration].append({
            "candidate": int(event.get("candidate") or 0),
            "valid": metrics["valid"],
            "violations": metrics["violations"],
            "excess_sum": metrics["excess_sum"],
            "distance": metrics["distance"],
            "crossings": metrics["crossings"],
            "reasons": metrics["reasons"],
            "route": event.get("route"),
        })
    for iteration in grouped:
        grouped[iteration].sort(key=lambda row: row["candidate"])
    return grouped


def _scorer_by_iteration(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "scorer_result":
            continue
        iteration = _event_iteration(event)
        if iteration is not None:
            result[iteration] = event
    return result


def _critic_scorer_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    critics = _critic_by_iteration(events)
    scorers = _scorer_by_iteration(events)
    iterations = sorted(set(critics) | set(scorers))
    rows: list[list[Any]] = []

    for iteration in iterations:
        candidates = critics.get(iteration, [])
        scorer = scorers.get(iteration)

        candidate_texts: list[str] = []
        valid_candidates: list[dict[str, Any]] = []
        by_id = {row["candidate"]: row for row in candidates}

        for row in candidates:
            if row["valid"] is True and row["distance"] is not None:
                valid_candidates.append(row)
            candidate_texts.append(
                f"C{row['candidate']}:"
                f"valid={_fmt(row['valid'])},"
                f"viol={row['violations']},"
                f"excess={_fmt(row['excess_sum'])},"
                f"dist={_fmt(row['distance'])},"
                f"cross={_fmt(row['crossings'])},"
                f"reason={_fmt(row['reasons'])}"
            )

        oracle = (
            min(valid_candidates, key=lambda row: float(row["distance"]))
            if valid_candidates else None
        )

        if scorer:
            best_id = int(scorer.get("best_id") or 0)
            selected = by_id.get(best_id)
            selected_valid = selected.get("valid") if selected else None
            selected_distance = selected.get("distance") if selected else None
            regret = None
            if (
                selected_valid is True
                and selected_distance is not None
                and oracle is not None
                and oracle.get("distance") is not None
            ):
                regret = float(selected_distance) - float(oracle["distance"])
            scorer_text = (
                f"order={_fmt(scorer.get('display_order') or [])}; "
                f"ranking={_fmt(scorer.get('ranking') or [])}; "
                f"best=C{best_id}; selected_valid={_fmt(selected_valid)}"
            )
        else:
            best_id = None
            regret = None
            scorer_text = "NOT REACHED"

        rows.append([
            iteration,
            len(candidates),
            sum(row["valid"] is True for row in candidates),
            " | ".join(candidate_texts) if candidate_texts else "-",
            scorer_text,
            f"C{oracle['candidate']} / {oracle['distance']}" if oracle else "-",
            regret,
        ])

    return rows


def _search_recovery_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for event in events:
        kind = event.get("event")
        scope = str(event.get("scope") or "")
        if scope.startswith("initializer"):
            continue
        if kind not in {"repair_result", "diversity_result", "hybrid_result", "restart_exhausted"}:
            continue
        evaluation = event.get("evaluation")
        metrics = _eval_metrics(evaluation) if isinstance(evaluation, dict) else {}
        rows.append([
            event.get("seq"),
            kind,
            scope or "-",
            _event_iteration(event),
            event.get("attempt"),
            event.get("restart_attempt"),
            metrics.get("valid"),
            metrics.get("violations"),
            metrics.get("excess_sum"),
            metrics.get("distance"),
            event.get("fallback_action"),
        ])
    return rows


def _error_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for event in events:
        if event.get("event") not in {
            "provider_error", "model_output_failure", "recoverable_agent_failure"
        }:
            continue
        rows.append([
            event.get("seq"),
            event.get("event"),
            event.get("phase"),
            event.get("scope"),
            _event_iteration(event),
            event.get("status_code"),
            event.get("error_type"),
            event.get("message") or event.get("error_message"),
        ])
    return rows


def _agent_cost_rows(calls: list[dict[str, Any]]) -> list[list[Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        grouped[str(row.get("agent") or "unknown")].append(row)

    rows: list[list[Any]] = []
    for agent in sorted(grouped):
        items = grouped[agent]
        rows.append([
            agent,
            len(items),
            sum(int(x.get("prompt_tokens") or 0) for x in items),
            sum(int(x.get("candidate_tokens") or 0) for x in items),
            sum(int(x.get("thought_tokens") or 0) for x in items),
            sum(int(x.get("total_tokens") or 0) for x in items),
            sum(float(x.get("latency") or 0.0) for x in items),
        ])
    return rows


def _best_failed_initializer(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["stage"] != "ACCEPTED" and row["valid"] is not True]
    if not candidates:
        return None

    def key(row: dict[str, Any]) -> tuple[float, float, int]:
        violations = row.get("violations")
        excess = row.get("excess_sum")
        return (
            float(violations) if violations is not None else 1e9,
            float(excess) if excess is not None else 1e9,
            int(row.get("seq") or 0),
        )

    return min(candidates, key=key)


def _build_report(run_root: Path, model_dir: Path) -> str:
    manifest = _read_json(run_root / "run.json")
    state = _read_json(model_dir / "state.json")
    events = _read_trace(model_dir / "trace.jsonl")
    calls = _call_rows(events)
    init_rows = _initializer_rows(events)
    critic_scorer = _critic_scorer_rows(events)
    recovery = _search_recovery_rows(events)
    errors = _error_rows(events)

    problem = manifest.get("problem") or {}
    config = manifest.get("config") or {}
    experiment = config.get("experiment") or {}
    render_policy = manifest.get("render_policy") or {}
    provider_policy = manifest.get("provider_policy") or {}

    initializer_accepted = any(event.get("event") == "initializer_result" for event in events)
    first_initializer = next((row for row in init_rows if row["stage"] == "Initializer"), None)
    best_failed = _best_failed_initializer(init_rows)

    lines: list[str] = [
        "AVMA-CVRP ANALYSIS REPORT",
        "=" * 140,
        "",
        "RUN SUMMARY",
        "-" * 140,
        f"Run ID               : {run_root.name}",
        f"Instance             : {_fmt(problem.get('name'))}",
        f"Dimension            : {_fmt(problem.get('dimension'))}",
        f"Provider / Model     : {model_dir.parent.name} / {model_dir.name}",
        f"Experiment           : {_fmt(experiment.get('name'))}",
        f"Target iterations    : {_fmt(experiment.get('iterations'))}",
        f"Completed iterations : {_fmt(state.get('completed_iterations') or 0)}",
        f"State status         : {_fmt(state.get('status'))}",
        f"Current phase        : {_fmt(state.get('current'))}",
        f"Reference optimum/BKS: {_fmt(problem.get('reference_optimum'))}",
        f"Demand encoding      : {_fmt(render_policy.get('demand_encoding_mode'))}",
        f"Bar layout           : {_fmt(render_policy.get('bar_layout'))}",
        f"Media resolution     : {_fmt(provider_policy.get('media_resolution'))}",
        "",
        "INITIALIZER SUMMARY",
        "-" * 140,
        f"Accepted             : {_fmt(initializer_accepted)}",
        f"First-shot valid     : {_fmt(first_initializer.get('valid') if first_initializer else None)}",
        f"Direct repairs       : {sum(row['stage'].startswith('Initializer Repair') for row in init_rows)}",
        f"Fallback restarts    : {sum(row['stage'].startswith('Fallback Restart') for row in init_rows)}",
        f"Fallback repairs     : {sum(row['stage'].startswith('Restart ') and ' Repair ' in row['stage'] for row in init_rows)}",
    ]

    if best_failed:
        lines.append(
            "Best failed attempt  : "
            f"seq={best_failed['seq']} / {best_failed['stage']} / "
            f"violations={best_failed['violations']} / "
            f"excess_sum={_fmt(best_failed['excess_sum'])} / "
            f"loads={_fmt(best_failed['loads'])}"
        )

    lines.extend([
        "",
        _table(
            "INITIALIZER OUTPUTS — chronological",
            [
                "Seq", "Stage", "Valid", "Viol", "ExcessΣ", "ExcessMax",
                "Route loads", "Capacity ratios", "Cross", "Reasons", "Route"
            ],
            [
                [
                    row["seq"], row["stage"], row["valid"], row["violations"],
                    row["excess_sum"], row["excess_max"], row["loads"], row["ratios"],
                    row["crossings"], row["reasons"], row["route"]
                ]
                for row in init_rows
            ],
            max_widths={1: 28, 6: 34, 7: 52, 9: 30, 10: 100},
        ),
        "",
        _table(
            "CRITIC + SCORER — per iteration",
            [
                "Iter", "Critic N", "Valid N", "Critic candidates",
                "Scorer", "Oracle best", "Selection regret"
            ],
            critic_scorer,
            max_widths={3: 115, 4: 90},
        ),
        "",
        _table(
            "SEARCH RECOVERY / ADAPTIVE EVENTS",
            ["Seq", "Event", "Scope", "Iter", "Attempt", "Restart", "Valid", "Viol", "ExcessΣ", "Distance", "Fallback"],
            recovery,
            max_widths={2: 55},
        ),
        "",
        _table(
            "AGENT COST",
            ["Agent", "Calls", "Prompt tok", "Output tok", "Thought tok", "Total tok", "Latency s"],
            _agent_cost_rows(calls),
        ),
        "",
        _table(
            "ERRORS / INTERRUPTIONS",
            ["Seq", "Event", "Phase", "Scope", "Iter", "Status", "Type", "Message"],
            errors,
            max_widths={2: 32, 3: 48, 7: 110},
        ),
        "",
        "STATUS FLAGS",
        "-" * 140,
        f"Initializer reached : {_fmt(bool(init_rows))}",
        f"Initializer accepted: {_fmt(initializer_accepted)}",
        f"Critic reached      : {_fmt(bool(_critic_by_iteration(events)))}",
        f"Scorer reached      : {_fmt(bool(_scorer_by_iteration(events)))}",
        f"Run completed       : {_fmt(state.get('status') == 'completed')}",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVMA-CVRP compact report-only analysis; partial initializer failures are supported."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    project = _find_project()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project / output_dir

    run_root = output_dir / "runs" / args.run_id
    if not run_root.is_dir():
        raise SystemExit(f"Run bulunamadı: {run_root}")

    model_dirs = sorted(path.parent for path in (run_root / "providers").glob("*/*/trace.jsonl"))
    if not model_dirs:
        model_dirs = sorted(path.parent for path in (run_root / "providers").glob("*/*/state.json"))

    if args.provider:
        model_dirs = [path for path in model_dirs if path.parent.name == args.provider]
    if args.model:
        model_dirs = [path for path in model_dirs if path.name == args.model]

    if not model_dirs:
        raise SystemExit("Filtreye uyan provider/model run klasörü bulunamadı.")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for model_dir in model_dirs:
        report = _build_report(run_root, model_dir)
        analysis_dir = model_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        report_path = analysis_dir / "report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"OK: {model_dir.parent.name}/{model_dir.name}")
        print(f"  {report_path}")


if __name__ == "__main__":
    main()
