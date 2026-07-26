"""eil51 için initializer, 7 critic adayı ve görsel scorer kullanan Multi-Agent 1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.core import (
    evaluate_route,
    method_dir,
    normalize_run_id,
    parse_tsplib,
    plot_route,
    read_json,
    write_json,
)
from src.gemini import (
    GEMINI_MODEL,
    parse_route,
    parse_scorer_response,
    request_candidates,
    request_scorer,
)
from src.metrics import elapsed_seconds, error_record, start_timer, summarize_api_calls
from src.summaries import multi_agent1_summary


ROOT = Path(__file__).resolve().parent
EVALUATION_KEYS = (
    "route",
    "validation",
    "legal_node_ids",
    "distance",
    "gap_to_known_optimum_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=ROOT / "data/eil51.tsp")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--run-id", default="eil51_run_01")
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def _solution(source: str, iteration: int, evaluation: dict, **extra: Any) -> dict:
    return {"source": source, "iteration": iteration, **extra, **evaluation}


def _evaluation_from(value: dict) -> dict:
    return {key: value[key] for key in EVALUATION_KEYS}


def _valid(values: list[dict]) -> list[dict]:
    return [
        value
        for value in values
        if value.get("validation", {}).get("is_valid")
        and value.get("distance") is not None
    ]


def _best_system(initializer: dict, iterations: list[dict]) -> dict | None:
    values = [initializer] + [
        _solution("scorer_selection", item["iteration"], _evaluation_from(item["selected_solution"]))
        for item in iterations
    ]
    valid = _valid(values)
    return min(valid, key=lambda value: value["distance"]) if valid else None


def _best_oracle(iterations: list[dict]) -> dict | None:
    values = [
        _solution(
            "critic_candidate_oracle",
            iteration["iteration"],
            _evaluation_from(candidate),
            candidate_id=candidate["candidate_id"],
        )
        for iteration in iterations
        for candidate in iteration["critic"]["candidates"]
    ]
    valid = _valid(values)
    return min(valid, key=lambda value: value["distance"]) if valid else None


def _calls(iterations: list[dict], pending: dict | None, errors: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for item in iterations:
        if item.get("critic", {}).get("api_call"):
            calls.append(item["critic"]["api_call"])
        if item.get("scorer", {}).get("api_call"):
            calls.append(item["scorer"]["api_call"])
    if pending:
        if pending.get("critic", {}).get("api_call"):
            calls.append(pending["critic"]["api_call"])
        calls.extend(
            attempt["api_call"]
            for attempt in pending.get("scorer_attempts", [])
            if attempt.get("api_call")
        )
    def identity(call: dict) -> tuple:
        return (
            call.get("phase"),
            call.get("started_at_utc"),
            call.get("finished_at_utc"),
        )

    known = {identity(call) for call in calls}
    for error in errors:
        call = error.get("api_call")
        if isinstance(call, dict) and identity(call) not in known:
            calls.append(call)
            known.add(identity(call))
    return calls


def _result(
    *,
    run_id: str,
    model: str,
    candidate_count: int,
    requested: int,
    initializer: dict,
    iterations: list[dict],
    pending: dict | None,
    errors: list[dict],
    invocation_seconds: float,
) -> dict:
    final = _evaluation_from(iterations[-1]["selected_solution"]) if iterations else initializer
    calls = _calls(iterations, pending, errors)
    return {
        "experiment": "gemini_visual_multi_agent_1_eil51",
        "run_id": run_id,
        "model": model,
        "num_locations_including_depot": 51,
        "num_salesmen": 1,
        "candidate_count_requested": candidate_count,
        "requested_iterations": requested,
        "completed_iterations": len(iterations),
        "artificial_delay_enabled": False,
        "scorer_policy": {
            "name": "feasibility_filtered_visual_scorer",
            "scope": "see_each_iteration_scorer_selection_mode",
            "python_validity_filter_enabled": True,
            "distance_or_gap_sent_to_scorer": False,
            "single_valid_candidate_is_selected_without_api": True,
            "no_valid_candidate_action": "retain_previous_route",
            "legacy_unfiltered_completed_iterations": [
                item["iteration"]
                for item in iterations
                if not item.get("scorer", {}).get("selection_mode")
            ],
            "filtered_completed_iterations": [
                item["iteration"]
                for item in iterations
                if item.get("scorer", {}).get("selection_mode")
            ],
        },
        "initializer": initializer,
        "iterations": iterations,
        "pending_iteration": pending,
        "final_solution": final,
        "best_valid_solution": _best_system(initializer, iterations),
        "best_critic_candidate_oracle": _best_oracle(iterations),
        "errors": errors,
        "run_summary": {
            **summarize_api_calls(calls),
            "current_invocation_wall_seconds_before_result_write": invocation_seconds,
        },
    }


def _save_checkpoint(
    path: Path,
    *,
    run_id: str,
    model: str,
    candidate_count: int,
    initializer: dict,
    iterations: list[dict],
    pending: dict | None,
    current_route: list[int],
    current_image: Path,
    errors: list[dict],
) -> None:
    write_json(
        path,
        {
            "run_id": run_id,
            "model": model,
            "candidate_count_requested": candidate_count,
            "initializer": initializer,
            "iterations": iterations,
            "pending_iteration": pending,
            "current_route": current_route,
            "current_image": str(current_image),
            "errors": errors,
        },
    )


def _finish_scorer(
    pending: dict,
    *,
    instance: Any,
    model: str,
    output: Path,
    fallback_route: list[int],
    fallback_image: Path,
) -> tuple[dict | None, Exception | None]:
    scorer_stage_timer = start_timer()
    all_candidates = pending["critic"]["candidates"]
    candidates = [
        candidate
        for candidate in all_candidates
        if candidate.get("validation", {}).get("is_valid") is True
    ]
    excluded_ids = [
        candidate["candidate_id"]
        for candidate in all_candidates
        if candidate not in candidates
    ]
    ids = [candidate["candidate_id"] for candidate in candidates]
    scores: dict[int, float] | None = None
    best_id: int | None = None
    raw_response: str | None = None
    scorer_call: dict | None = None
    reused_stored_response = False
    parsing_seconds = 0.0
    selection_mode = "visual_scorer_after_feasibility_filter"

    if not candidates:
        evaluation = evaluate_route(instance, fallback_route)
        iteration = pending["iteration"]
        selected_image = output / "images" / f"iteration_{iteration:02d}" / "selected.png"
        rendering_timer = start_timer()
        plot_route(
            instance,
            fallback_route,
            selected_image,
            title=f"eil51 Multi-Agent 1 — iteration {iteration} retained previous route",
        )
        rendering_seconds = elapsed_seconds(rendering_timer)
        return (
            {
                "iteration": iteration,
                "iteration_type": "critic_candidates_then_feasibility_filtered_scorer",
                "critic": pending["critic"],
                "scorer": {
                    "temperature": None,
                    "selection_mode": "retain_previous_route_no_valid_candidate",
                    "eligible_candidate_ids": [],
                    "excluded_invalid_candidate_ids": excluded_ids,
                    "raw_response": None,
                    "scores": {},
                    "best_candidate_id": None,
                    "selection_regret_percent_after_evaluation": None,
                    "api_call": None,
                    "attempt_count": len(pending.get("scorer_attempts", [])),
                    "timing": {
                        "reused_stored_response": False,
                        "response_parsing_seconds": 0.0,
                        "selected_route_rendering_seconds": rendering_seconds,
                        "scorer_stage_wall_seconds": elapsed_seconds(scorer_stage_timer),
                    },
                },
                "selected_solution": {
                    **evaluation,
                    "route_image": str(selected_image),
                    "retained_from_image": str(fallback_image),
                },
            },
            None,
        )

    if len(candidates) == 1:
        best_id = ids[0]
        scores = {}
        selection_mode = "single_valid_candidate_without_api"

    for attempt in reversed(pending.get("scorer_attempts", [])) if best_id is None else []:
        try:
            parsing_timer = start_timer()
            scores, best_id = parse_scorer_response(
                attempt["raw_response"], expected_image_ids=ids
            )
            parsing_seconds += elapsed_seconds(parsing_timer)
            raw_response = attempt["raw_response"]
            scorer_call = attempt.get("api_call")
            reused_stored_response = True
            break
        except Exception:
            parsing_seconds += elapsed_seconds(parsing_timer)
            continue

    if best_id is None:
        try:
            response = request_scorer(
                [Path(candidate["route_image"]) for candidate in candidates],
                image_ids=ids,
                model=model,
            )
            raw_response = response.text
            scorer_call = response.api_call
            pending.setdefault("scorer_attempts", []).append(
                {"raw_response": raw_response, "api_call": scorer_call}
            )
            parsing_timer = start_timer()
            scores, best_id = parse_scorer_response(
                raw_response, expected_image_ids=ids
            )
            parsing_seconds += elapsed_seconds(parsing_timer)
        except Exception as exc:
            if raw_response is not None and not pending.get("scorer_attempts"):
                pending.setdefault("scorer_attempts", []).append(
                    {"raw_response": raw_response, "api_call": scorer_call}
                )
            if scorer_call is not None and not hasattr(exc, "gemini_call_record"):
                try:
                    setattr(exc, "gemini_call_record", scorer_call)
                except Exception:
                    pass
            return None, exc

    selected = next(item for item in candidates if item["candidate_id"] == best_id)
    valid = _valid(candidates)
    best_distance = min(item["distance"] for item in valid) if valid else None
    selected_distance = selected.get("distance")
    regret = (
        100.0 * (selected_distance - best_distance) / best_distance
        if selected_distance is not None and best_distance
        else None
    )
    iteration = pending["iteration"]
    rendering_timer = start_timer()
    selected_image = output / "images" / f"iteration_{iteration:02d}" / "selected.png"
    plot_route(
        instance,
        selected["route"],
        selected_image,
        title=f"eil51 Multi-Agent 1 — iteration {iteration} selected {best_id}",
    )
    rendering_seconds = elapsed_seconds(rendering_timer)
    selected_solution = {**_evaluation_from(selected), "route_image": str(selected_image)}
    return (
        {
            "iteration": iteration,
            "iteration_type": "critic_candidates_then_feasibility_filtered_scorer",
            "critic": pending["critic"],
            "scorer": {
                "temperature": (
                    0.0
                    if selection_mode == "visual_scorer_after_feasibility_filter"
                    else None
                ),
                "selection_mode": selection_mode,
                "eligible_candidate_ids": ids,
                "excluded_invalid_candidate_ids": excluded_ids,
                "raw_response": raw_response,
                "scores": {str(key): value for key, value in (scores or {}).items()},
                "best_candidate_id": best_id,
                "selection_regret_percent_after_evaluation": regret,
                "api_call": scorer_call,
                "attempt_count": len(pending.get("scorer_attempts", [])),
                "timing": {
                    "reused_stored_response": reused_stored_response,
                    "response_parsing_seconds": parsing_seconds,
                    "selected_route_rendering_seconds": rendering_seconds,
                    "scorer_stage_wall_seconds": elapsed_seconds(scorer_stage_timer),
                },
            },
            "selected_solution": selected_solution,
        },
        None,
    )


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or not 1 <= args.candidate_count <= 7:
        raise SystemExit("iterations >= 1 ve candidate-count 1..7 olmalıdır.")
    run_id = normalize_run_id(args.run_id)
    output = method_dir(args.output_dir, run_id, "multi_agent1")
    result_path = output / "gemini_multi_agent1_results.json"
    summary_path = output / "gemini_multi_agent1_summary.json"
    checkpoint_path = output / "gemini_multi_agent1_checkpoint.json"
    if args.summary_only:
        write_json(summary_path, multi_agent1_summary(read_json(result_path)))
        print(f"Özet dosyası: {summary_path}")
        return

    instance = parse_tsplib(args.instance)
    zero_path = method_dir(args.output_dir, run_id, "zero_shot") / "gemini_zero_shot_results.json"
    if not zero_path.exists():
        raise SystemExit("Önce aynı --run-id ile zero-shot deneyi çalıştırılmalıdır.")
    zero = read_json(zero_path)
    zero_route = [int(value) for value in zero.get("route", [])]
    initializer = _solution("zero_shot", 0, evaluate_route(instance, zero_route))
    initializer["source_file"] = str(zero_path)
    current_image = Path(zero.get("route_image", ""))
    current_route = zero_route
    if not current_image.exists():
        raise SystemExit(f"Zero-shot rota görseli bulunamadı: {current_image}")

    if args.validate_only:
        print("Multi-Agent 1 çevrimdışı doğrulaması başarılı.")
        print(f"Model: {args.model}")
        print(f"Nokta sayısı: {instance.dimension}")
        print(f"Hedef iterasyon: {args.iterations}")
        print(f"Critic adayı / iterasyon: {args.candidate_count}")
        print(f"Tahmini Gemini isteği: {args.iterations * 2}")
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
        return

    invocation_timer = start_timer()
    iterations: list[dict] = []
    errors: list[dict] = []
    pending: dict | None = None
    if args.resume and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        if (
            checkpoint.get("run_id") != run_id
            or checkpoint.get("model") != args.model
            or checkpoint.get("candidate_count_requested") != args.candidate_count
        ):
            raise SystemExit("Checkpoint run-id/model/candidate-count ile uyuşmuyor.")
        iterations = checkpoint.get("iterations", [])
        errors = checkpoint.get("errors", [])
        pending = checkpoint.get("pending_iteration")
        current_route = [int(value) for value in checkpoint["current_route"]]
        current_image = Path(checkpoint["current_image"])
        print(
            f"Checkpoint yüklendi: {len(iterations)} tam iterasyon, "
            f"bekleyen scorer aşaması={'var' if pending else 'yok'}."
        )
    elif args.resume:
        print("Checkpoint bulunamadı; deney baştan başlatılıyor.")

    stopped = False
    next_iteration = len(iterations) + 1
    while next_iteration <= args.iterations and not stopped:
        if pending:
            print(
                f"\nİterasyon {pending['iteration']}: kayıtlı critic adayları "
                "yeniden kullanılacak; yalnız scorer çağrılacak."
            )
        else:
            print(
                f"\n=== Multi-Agent 1 iterasyon {next_iteration}: "
                f"{args.candidate_count} critic adayı ==="
            )
            try:
                critic_stage_timer = start_timer()
                response = request_candidates(
                    current_image,
                    candidate_count=args.candidate_count,
                    model=args.model,
                    temperature=0.7,
                )
                candidates: list[dict] = []
                for candidate_id, raw in enumerate(response.texts, start=1):
                    route = parse_route(raw)
                    evaluation = evaluate_route(instance, route)
                    image = (
                        output
                        / "images"
                        / f"iteration_{next_iteration:02d}"
                        / f"candidate_{candidate_id:02d}.png"
                    )
                    plot_route(
                        instance,
                        route,
                        image,
                        title=(
                            f"eil51 MA1 iteration {next_iteration} "
                            f"candidate {candidate_id}"
                        ),
                    )
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "raw_response": raw,
                            **evaluation,
                            "route_image": str(image),
                        }
                    )
                    print(
                        f"Aday {candidate_id}: geçerli={evaluation['validation']['is_valid']}, "
                        f"mesafe={evaluation['distance']}, "
                        f"gap={evaluation['gap_to_known_optimum_percent']}"
                    )
                pending = {
                    "iteration": next_iteration,
                    "critic": {
                        "temperature": 0.7,
                        "requested_candidate_count": args.candidate_count,
                        "returned_candidate_count": len(candidates),
                        "input_image": str(current_image),
                        "api_call": response.api_call,
                        "timing": {
                            "critic_stage_wall_seconds": elapsed_seconds(
                                critic_stage_timer
                            ),
                            "api_call_wall_seconds": response.api_call[
                                "api_call_wall_seconds"
                            ],
                        },
                        "candidates": candidates,
                    },
                    "scorer_attempts": [],
                }
                _save_checkpoint(
                    checkpoint_path,
                    run_id=run_id,
                    model=args.model,
                    candidate_count=args.candidate_count,
                    initializer=initializer,
                    iterations=iterations,
                    pending=pending,
                    current_route=current_route,
                    current_image=current_image,
                    errors=errors,
                )
            except Exception as exc:
                errors.append(
                    error_record(exc, phase="critic_candidate_generation", iteration=next_iteration)
                )
                print(f"Critic iterasyon {next_iteration} tamamlanamadı: {exc}")
                stopped = True
                break

        completed, scorer_error = _finish_scorer(
            pending,
            instance=instance,
            model=args.model,
            output=output,
            fallback_route=current_route,
            fallback_image=current_image,
        )
        if scorer_error is not None:
            errors.append(
                error_record(
                    scorer_error,
                    phase="visual_scorer",
                    iteration=pending["iteration"],
                )
            )
            _save_checkpoint(
                checkpoint_path,
                run_id=run_id,
                model=args.model,
                candidate_count=args.candidate_count,
                initializer=initializer,
                iterations=iterations,
                pending=pending,
                current_route=current_route,
                current_image=current_image,
                errors=errors,
            )
            print(f"Scorer iterasyon {pending['iteration']} tamamlanamadı: {scorer_error}")
            stopped = True
            break

        assert completed is not None
        iterations.append(completed)
        selected = completed["selected_solution"]
        current_route = selected["route"]
        current_image = Path(selected["route_image"])
        best_id = completed["scorer"]["best_candidate_id"]
        if best_id is None:
            print("Geçerli critic adayı yok; önceki rota korundu.")
        elif completed["scorer"]["selection_mode"] == "single_valid_candidate_without_api":
            print(f"Tek geçerli aday otomatik seçildi: aday {best_id}")
        else:
            print(f"Scorer seçimi: aday {best_id}")
        print(f"Seçilen mesafe: {selected['distance']}")
        pending = None
        next_iteration = len(iterations) + 1
        _save_checkpoint(
            checkpoint_path,
            run_id=run_id,
            model=args.model,
            candidate_count=args.candidate_count,
            initializer=initializer,
            iterations=iterations,
            pending=pending,
            current_route=current_route,
            current_image=current_image,
            errors=errors,
        )

    result = _result(
        run_id=run_id,
        model=args.model,
        candidate_count=args.candidate_count,
        requested=args.iterations,
        initializer=initializer,
        iterations=iterations,
        pending=pending,
        errors=errors,
        invocation_seconds=elapsed_seconds(invocation_timer),
    )
    write_json(result_path, result)
    write_json(summary_path, multi_agent1_summary(result))
    print("\nGemini eil51 Multi-Agent 1 durumu kaydedildi.")
    print(f"Tamamlanan tam iterasyon: {len(iterations)}")
    if pending:
        print(f"Scorer bekleyen iterasyon: {pending['iteration']} (--resume ile devam edilir)")
    print(f"Sonuç dosyası: {result_path}")
    print(f"Özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
