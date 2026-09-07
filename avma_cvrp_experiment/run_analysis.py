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
    raise SystemExit("avma_cvrp_experiment bulunamadı")


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
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path} satır {lineno} JSON değil: {exc}") from exc
            if isinstance(obj, dict):
                events.append(obj)
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


def _fmt_gap(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}%"


def _gap(distance: Any, reference_optimum: float | None) -> float | None:
    if distance is None or reference_optimum is None or reference_optimum <= 0:
        return None
    return ((float(distance) - reference_optimum) / reference_optimum) * 100.0


def _truncate(text: str, limit: int) -> str:
    text = str(text).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _table(title: str, headers: list[str], rows: list[list[Any]], *, max_widths: dict[int, int] | None = None) -> str:
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
        out.append(line(["(no records)"] + [""] * (len(headers) - 1)))
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


def _metrics(evaluation: dict[str, Any] | None) -> dict[str, Any]:
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
        "loads": validation.get("route_loads") or [],
        "ratios": ratios_raw,
        "violations": len(exceeded),
        "excess_sum": sum(excesses) if ratios else None,
        "excess_max": max(excesses) if ratios else None,
        "distance": evaluation.get("distance"),
        "crossings": evaluation.get("crossings"),
    }


def _calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if event.get("event") != "agent_call":
            continue
        call = event.get("call") or {}
        usage = call.get("usage") or {}
        rows.append({
            "agent": event.get("agent") or "unknown",
            "latency": float(call.get("latency_seconds") or 0.0),
            "prompt": int(usage.get("prompt_token_count") or 0),
            "output": int(usage.get("candidates_token_count") or 0),
            "thought": int(usage.get("thoughts_token_count") or 0),
            "total": int(usage.get("total_token_count") or 0),
        })
    return rows


def _initializer_rows(events: list[dict[str, Any]], ref: float | None) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        kind = event.get("event")
        scope = str(event.get("scope") or "")
        stage = route = evaluation = None
        if kind == "initializer_candidate":
            stage, route, evaluation = "Initializer", event.get("route"), event.get("evaluation")
        elif kind == "repair_result" and scope == "initializer":
            stage, route, evaluation = f"Initializer Repair {event.get('attempt') or '?'}", event.get("output_route"), event.get("evaluation")
        elif kind == "diversity_result" and scope == "initializer.fallback":
            stage, route, evaluation = f"Fallback Restart {event.get('restart_attempt') or '?'}", event.get("route"), event.get("evaluation")
        elif kind == "initializer_result":
            stage, route, evaluation = "ACCEPTED", event.get("accepted_route"), event.get("evaluation")
        if stage is None:
            continue
        m = _metrics(evaluation)
        rows.append({"seq": event.get("seq"), "stage": stage, "route": route, **m, "gap": _gap(m["distance"], ref) if m["valid"] is True else None})
    return rows


def _critics(events: list[dict[str, Any]], ref: float | None) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("event") != "critic_result":
            continue
        iteration = _event_iteration(event)
        if iteration is None:
            continue
        m = _metrics(event.get("evaluation"))
        grouped[iteration].append({
            "seq": event.get("seq"),
            "candidate": int(event.get("candidate") or 0),
            **m,
            "gap": _gap(m["distance"], ref) if m["valid"] is True else None,
        })
    for iteration in grouped:
        grouped[iteration].sort(key=lambda row: row["candidate"])
    return grouped


def _scorers(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {}
    for event in events:
        if event.get("event") == "scorer_result":
            iteration = _event_iteration(event)
            if iteration is not None:
                result[iteration] = event
    return result


def _working_by_iteration(events: list[dict[str, Any]]) -> dict[int, float]:
    result = {}
    for event in events:
        if event.get("event") != "iteration_result":
            continue
        iteration = _event_iteration(event)
        payload = event.get("result") or {}
        m = _metrics(payload.get("working_evaluation"))
        if iteration is not None and m["valid"] is True and m["distance"] is not None:
            result[iteration] = float(m["distance"])
    return result


def _accepted_initializer(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        if event.get("event") == "initializer_result":
            m = _metrics(event.get("evaluation"))
            if m["valid"] is True and m["distance"] is not None:
                return float(m["distance"])
    return None


def _progress(events: list[dict[str, Any]], ref: float | None) -> list[dict[str, Any]]:
    critics = _critics(events, ref)
    working = _working_by_iteration(events)
    init = _accepted_initializer(events)
    observed_best = selected_best = init
    rows = [{"stage": "Initializer", "iteration": 0, "observed": observed_best, "selected": selected_best}]
    for iteration in sorted(set(critics) | set(working)):
        valid = [float(r["distance"]) for r in critics.get(iteration, []) if r["valid"] is True and r["distance"] is not None]
        if valid:
            observed_best = min([x for x in [observed_best, min(valid)] if x is not None])
        if iteration in working:
            selected_best = min([x for x in [selected_best, working[iteration]] if x is not None])
        rows.append({
            "stage": f"Iter {iteration}{'*' if iteration not in working else ''}",
            "iteration": iteration,
            "observed": observed_best,
            "selected": selected_best,
        })
    return rows


def _critic_table(events: list[dict[str, Any]], ref: float | None) -> list[list[Any]]:
    critics = _critics(events, ref)
    scorers = _scorers(events)
    progress = {row["iteration"]: row for row in _progress(events, ref)}
    rows = []
    for iteration in sorted(set(critics) | set(scorers)):
        candidates = critics.get(iteration, [])
        valid = [row for row in candidates if row["valid"] is True and row["distance"] is not None]
        oracle = min(valid, key=lambda row: float(row["distance"])) if valid else None
        by_id = {row["candidate"]: row for row in candidates}
        scorer = scorers.get(iteration)
        selected_distance = selected_gap = regret = None
        scorer_text = "NOT REACHED"
        if scorer:
            best_id = int(scorer.get("best_id") or 0)
            selected = by_id.get(best_id)
            if selected and selected["valid"] is True and selected["distance"] is not None:
                selected_distance = float(selected["distance"])
                selected_gap = _gap(selected_distance, ref)
                if oracle is not None:
                    regret = selected_distance - float(oracle["distance"])
            scorer_text = f"order={_fmt(scorer.get('display_order') or [])}; ranking={_fmt(scorer.get('ranking') or [])}; best=C{best_id}"
        candidate_text = " | ".join(
            f"C{r['candidate']}:valid={_fmt(r['valid'])},viol={r['violations']},excess={_fmt(r['excess_sum'])},dist={_fmt(r['distance'])},gap={_fmt_gap(r['gap'])},cross={_fmt(r['crossings'])},reason={_fmt(r['reasons'])}"
            for r in candidates
        ) or "-"
        p = progress.get(iteration) or {}
        rows.append([
            iteration,
            len(candidates),
            len(valid),
            float(oracle["distance"]) if oracle else None,
            _fmt_gap(_gap(oracle["distance"], ref)) if oracle else "-",
            selected_distance,
            _fmt_gap(selected_gap),
            p.get("observed"),
            p.get("selected"),
            regret,
            candidate_text,
            scorer_text,
        ])
    return rows


def _recovery_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for event in events:
        kind = event.get("event")
        scope = str(event.get("scope") or "")
        if scope.startswith("initializer") or kind not in {"repair_result", "diversity_result", "hybrid_result", "restart_exhausted"}:
            continue
        m = _metrics(event.get("evaluation") if isinstance(event.get("evaluation"), dict) else {})
        rows.append([event.get("seq"), kind, scope or "-", _event_iteration(event), event.get("attempt"), event.get("restart_attempt"), m["valid"], m["violations"], m["excess_sum"], m["distance"], event.get("fallback_action")])
    return rows


def _error_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for event in events:
        if event.get("event") in {"provider_error", "model_output_failure", "recoverable_agent_failure"}:
            rows.append([event.get("seq"), event.get("event"), event.get("phase"), event.get("scope"), _event_iteration(event), event.get("status_code"), event.get("error_type"), event.get("message") or event.get("error_message")])
    return rows


def _agent_cost_rows(calls: list[dict[str, Any]]) -> list[list[Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        grouped[str(row["agent"])].append(row)
    return [[agent, len(items), sum(x["prompt"] for x in items), sum(x["output"] for x in items), sum(x["thought"] for x in items), sum(x["total"] for x in items), sum(x["latency"] for x in items)] for agent, items in sorted(grouped.items())]


def _build_report(run_root: Path, model_dir: Path, ref_override: float | None) -> tuple[str, list[dict[str, Any]], float | None]:
    manifest = _read_json(run_root / "run.json")
    state = _read_json(model_dir / "state.json")
    events = _read_trace(model_dir / "trace.jsonl")
    calls = _calls(events)
    problem = manifest.get("problem") or {}
    manifest_ref = problem.get("reference_optimum")
    ref = float(ref_override) if ref_override is not None else (float(manifest_ref) if manifest_ref is not None else None)
    init_rows = _initializer_rows(events, ref)
    progress = _progress(events, ref)
    last = progress[-1] if progress else {}
    observer_gbest = last.get("observed")
    selected_gbest = last.get("selected")
    config = manifest.get("config") or {}
    experiment = config.get("experiment") or {}
    render_policy = manifest.get("render_policy") or {}
    provider_policy = manifest.get("provider_policy") or {}
    first_initializer = next((row for row in init_rows if row["stage"] == "Initializer"), None)
    best_failed = min((row for row in init_rows if row["stage"] != "ACCEPTED" and row["valid"] is not True), key=lambda r: (r["violations"], r["excess_sum"] if r["excess_sum"] is not None else 1e9, r["seq"] or 0), default=None)
    final_distance = selected_gbest if state.get("status") == "completed" else None
    total = {
        "calls": len(calls),
        "prompt": sum(x["prompt"] for x in calls),
        "output": sum(x["output"] for x in calls),
        "thought": sum(x["thought"] for x in calls),
        "tokens": sum(x["total"] for x in calls),
        "latency": sum(x["latency"] for x in calls),
    }
    lines = [
        "AVMA-CVRP ANALYSIS REPORT", "=" * 140, "", "RUN SUMMARY", "-" * 140,
        f"Run ID                 : {run_root.name}",
        f"Instance               : {_fmt(problem.get('name'))}",
        f"Dimension              : {_fmt(problem.get('dimension'))}",
        f"Provider / Model       : {model_dir.parent.name} / {model_dir.name}",
        f"Experiment             : {_fmt(experiment.get('name'))}",
        f"Target iterations      : {_fmt(experiment.get('iterations'))}",
        f"Completed iterations   : {_fmt(state.get('completed_iterations') or 0)}",
        f"State status           : {_fmt(state.get('status'))}",
        f"Current phase          : {_fmt(state.get('current'))}",
        f"Reference optimum/BKS  : {_fmt(ref)}",
        f"Demand encoding        : {_fmt(render_policy.get('demand_encoding_mode'))}",
        f"Bar layout             : {_fmt(render_policy.get('bar_layout'))}",
        f"Media resolution       : {_fmt(provider_policy.get('media_resolution'))}", "",
        f"Observer GBest         : {_fmt(observer_gbest)}",
        f"Observer GBest gap     : {_fmt_gap(_gap(observer_gbest, ref))}",
        f"Selected GBest         : {_fmt(selected_gbest)}",
        f"Selected GBest gap     : {_fmt_gap(_gap(selected_gbest, ref))}",
        f"Final distance         : {_fmt(final_distance)}",
        f"Final gap              : {_fmt_gap(_gap(final_distance, ref))}",
        f"Run completed          : {_fmt(state.get('status') == 'completed')}", "",
        "INITIALIZER SUMMARY", "-" * 140,
        f"Accepted               : {_fmt(any(e.get('event') == 'initializer_result' for e in events))}",
        f"First-shot valid       : {_fmt(first_initializer.get('valid') if first_initializer else None)}",
        f"Direct repairs         : {sum(r['stage'].startswith('Initializer Repair') for r in init_rows)}",
        f"Fallback restarts      : {sum(r['stage'].startswith('Fallback Restart') for r in init_rows)}",
    ]
    if best_failed:
        lines.append(f"Best failed attempt    : seq={best_failed['seq']} / {best_failed['stage']} / violations={best_failed['violations']} / excess_sum={_fmt(best_failed['excess_sum'])} / loads={_fmt(best_failed['loads'])}")
    lines += [
        "",
        _table("INITIALIZER OUTPUTS — chronological", ["Seq", "Stage", "Valid", "Viol", "ExcessΣ", "ExcessMax", "Route loads", "Capacity ratios", "Distance", "Gap %", "Cross", "Reasons", "Route"], [[r["seq"], r["stage"], r["valid"], r["violations"], r["excess_sum"], r["excess_max"], r["loads"], r["ratios"], r["distance"], _fmt_gap(r["gap"]), r["crossings"], r["reasons"], r["route"]] for r in init_rows], max_widths={1: 28, 6: 34, 7: 52, 11: 30, 12: 100}),
        "",
        _table("CRITIC + SCORER — per iteration", ["Iter", "Critic N", "Valid N", "Iter Best", "Iter Gap %", "Selected", "Sel Gap %", "Observer GBest", "Selected GBest", "Regret", "Critic candidates", "Scorer"], _critic_table(events, ref), max_widths={10: 120, 11: 85}),
        "",
        _table("SEARCH RECOVERY / ADAPTIVE EVENTS", ["Seq", "Event", "Scope", "Iter", "Attempt", "Restart", "Valid", "Viol", "ExcessΣ", "Distance", "Fallback"], _recovery_rows(events), max_widths={2: 55}),
        "",
        _table("AGENT COST", ["Agent", "Calls", "Prompt tok", "Output tok", "Thought tok", "Total tok", "Latency s"], _agent_cost_rows(calls)),
        "", "TOTAL COST / USAGE", "-" * 140,
        f"Total API calls        : {total['calls']}", f"Prompt tokens          : {total['prompt']}", f"Output tokens          : {total['output']}", f"Thinking tokens        : {total['thought']}", f"Total tokens           : {total['tokens']}", f"Total API latency      : {total['latency']:.4f} s", "",
        _table("ERRORS / INTERRUPTIONS", ["Seq", "Event", "Phase", "Scope", "Iter", "Status", "Type", "Message"], _error_rows(events), max_widths={2: 32, 3: 48, 7: 110}),
        "", "STATUS FLAGS", "-" * 140,
        f"Initializer reached : {_fmt(bool(init_rows))}", f"Initializer accepted: {_fmt(any(e.get('event') == 'initializer_result' for e in events))}", f"Critic reached      : {_fmt(bool(_critics(events, ref)))}", f"Scorer reached      : {_fmt(bool(_scorers(events)))}", f"Run completed       : {_fmt(state.get('status') == 'completed')}", "",
    ]
    return "\n".join(lines), progress, ref


def _write_chart(analysis_dir: Path, progress: list[dict[str, Any]], ref: float | None, run_id: str) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Grafik üretmek için matplotlib gerekli: pip install matplotlib") from exc
    usable = [row for row in progress if row.get("observed") is not None or row.get("selected") is not None]
    if not usable:
        raise SystemExit("Grafik için geçerli distance kaydı bulunamadı")
    labels = [str(row["stage"]) for row in usable]
    x = list(range(len(labels)))
    plt.figure(figsize=(11, 6))
    plt.plot(x, [row.get("observed") for row in usable], marker="o", label="Observer GBest")
    plt.plot(x, [row.get("selected") for row in usable], marker="o", label="Selected GBest")
    if ref is not None:
        plt.axhline(ref, linestyle="--", label=f"BKS / optimum = {ref:g}")
    plt.xticks(x, labels)
    plt.xlabel("Search stage")
    plt.ylabel("CVRP distance")
    plt.title(f"{run_id} — search progress")
    plt.legend()
    plt.tight_layout()
    path = analysis_dir / "search_progress.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="AVMA-CVRP report + GBest/gap progress chart")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reference-optimum", type=float, default=None, help="Analysis-only BKS/optimum override")
    args = parser.parse_args()
    if args.reference_optimum is not None and args.reference_optimum <= 0:
        raise SystemExit("--reference-optimum pozitif olmalıdır")
    project = _find_project()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
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
        raise SystemExit("Filtreye uyan provider/model run klasörü bulunamadı")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for model_dir in model_dirs:
        report, progress, ref = _build_report(run_root, model_dir, args.reference_optimum)
        analysis_dir = model_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        report_path = analysis_dir / "report.txt"
        report_path.write_text(report, encoding="utf-8")
        chart_path = _write_chart(analysis_dir, progress, ref, run_root.name)
        print(f"OK: {model_dir.parent.name}/{model_dir.name}")
        print(f"  Report : {report_path}")
        print(f"  Chart  : {chart_path}")


if __name__ == "__main__":
    main()
