"""OpenRouter modelleri için critic ve görsel scorer Multi-Agent 1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.core import (
    evaluate_route,
    normalize_run_id,
    plot_route,
    read_json,
    write_json,
)
from src.gemini import (
    parse_route,
    parse_scorer_response,
)
from src.metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
)
from src.openrouter import (
    OPENROUTER_MODELS,
    request_candidates,
    request_scorer,
    resolve_model_alias,
)
from src.problem_instance import ProblemInstance
from src.run_manifest import load_run_problem


ROOT = Path(__file__).resolve().parent
EVALUATION_KEYS = (
    "route",
    "validation",
    "legal_node_ids",
    "distance",
    "reference_distance",
    "gap_to_reference_percent",
)


def _model_root(run_dir: Path, alias: str) -> Path:
    return (
        run_dir
        / "model_comparisons"
        / "openrouter"
        / alias
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(OPENROUTER_MODELS),
        help="OpenRouter modelinin kısa deney adı.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=7)
    parser.add_argument(
        "--candidate-strategy",
        choices=(
            "independent_calls",
            "native_multiple_choices",
        ),
        default="independent_calls",
        help=(
            "Varsayılan independent_calls, sağlayıcı n değerini "
            "yok saysa bile istenen aday sayısını üretir."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Girdileri API çağrısı yapmadan doğrular.",
    )
    return parser.parse_args()


def _evaluation_from(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in EVALUATION_KEYS
    }


def _solution(
    source: str,
    iteration: int,
    evaluation: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "source": source,
        "iteration": iteration,
        **extra,
        **_evaluation_from(evaluation),
    }


def _valid(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        value
        for value in values
        if (
            value.get("validation", {}).get("is_valid") is True
            and value.get("distance") is not None
        )
    ]


def _best_system(
    initializer: dict[str, Any],
    iterations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    values = [initializer] + [
        _solution(
            "scorer_selection",
            item["iteration"],
            item["selected_solution"],
        )
        for item in iterations
    ]
    valid = _valid(values)
    return (
        min(valid, key=lambda value: value["distance"])
        if valid
        else None
    )


def _best_oracle(
    iterations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    values = [
        _solution(
            "critic_candidate_oracle",
            iteration["iteration"],
            candidate,
            candidate_id=candidate["candidate_id"],
        )
        for iteration in iterations
        for candidate in iteration["critic"]["candidates"]
    ]
    valid = _valid(values)
    return (
        min(valid, key=lambda value: value["distance"])
        if valid
        else None
    )


def _relative(path: Path, run_dir: Path) -> str:
    return path.resolve().relative_to(
        run_dir.resolve()
    ).as_posix()


def _resolve_run_artifact(
    run_dir: Path,
    relative_path: str,
) -> Path:
    path = (run_dir / relative_path).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Run klasörü dışındaki artifact reddedildi: {relative_path}"
        ) from exc
    return path


def _calls(
    iterations: list[dict[str, Any]],
    pending: dict[str, Any] | None,
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def extend_critic(value: dict[str, Any]) -> None:
        critic_calls = value.get("api_calls")
        if isinstance(critic_calls, list):
            calls.extend(
                call
                for call in critic_calls
                if isinstance(call, dict)
            )
            return
        critic_call = value.get("api_call")
        if isinstance(critic_call, dict):
            calls.append(critic_call)

    for item in iterations:
        extend_critic(item.get("critic", {}))
        scorer_call = item.get("scorer", {}).get("api_call")
        if isinstance(scorer_call, dict):
            calls.append(scorer_call)
    if pending:
        extend_critic(pending.get("critic", {}))
        for attempt in pending.get("scorer_attempts", []):
            call = attempt.get("api_call")
            if isinstance(call, dict):
                calls.append(call)

    def identity(call: dict[str, Any]) -> tuple[Any, ...]:
        return (
            call.get("phase"),
            call.get("started_at_utc"),
            call.get("finished_at_utc"),
        )

    known = {
        identity(call)
        for call in calls
    }
    for error in errors:
        error_calls = error.get("api_calls")
        if not isinstance(error_calls, list):
            error_calls = [error.get("api_call")]
        for call in error_calls:
            if (
                isinstance(call, dict)
                and identity(call) not in known
            ):
                calls.append(call)
                known.add(identity(call))
    return calls


def _result(
    *,
    run_id: str,
    problem: ProblemInstance,
    fingerprint: str,
    model_alias: str,
    model: str,
    candidate_count: int,
    candidate_strategy: str,
    requested: int,
    initializer: dict[str, Any],
    iterations: list[dict[str, Any]],
    pending: dict[str, Any] | None,
    errors: list[dict[str, Any]],
    loading_seconds: float,
    invocation_seconds: float,
) -> dict[str, Any]:
    final = (
        _solution(
            "scorer_selection",
            iterations[-1]["iteration"],
            iterations[-1]["selected_solution"],
        )
        if iterations
        else initializer
    )
    calls = _calls(iterations, pending, errors)
    return {
        "schema_version": "2.0",
        "experiment": "dynamic_openrouter_multi_agent_1_tsp",
        "run_id": run_id,
        "method": "multi_agent_1",
        "problem": {
            "name": problem.name,
            "dimension": problem.dimension,
            "depot_id": problem.depot_id,
            "source_type": problem.source_type.value,
            "fingerprint_sha256": fingerprint,
            "reference_type": (
                problem.reference.reference_type.value
                if problem.reference is not None
                else None
            ),
            "reference_is_proven_optimal": (
                problem.reference.is_proven_optimal
                if problem.reference is not None
                else None
            ),
        },
        "model": {
            "provider": "openrouter",
            "alias": model_alias,
            "requested_name": model,
            "critic_temperature": 0.7,
            "scorer_temperature": 0.0,
            "reasoning_effort": "none",
        },
        "critic_candidate_generation": {
            "strategy": candidate_strategy,
            "single_http_request_per_iteration": (
                candidate_strategy
                == "native_multiple_choices"
            ),
            "requested_choice_count": candidate_count,
        },
        "candidate_count_requested": candidate_count,
        "requested_iterations": requested,
        "completed_iterations": len(iterations),
        "artificial_delay_enabled": False,
        "scorer_policy": {
            "name": "feasibility_filtered_visual_scorer",
            "python_validity_filter_enabled": True,
            "distance_or_gap_sent_to_scorer": False,
            "single_valid_candidate_is_selected_without_api": True,
            "no_valid_candidate_action": "retain_previous_route",
        },
        "initializer": initializer,
        "iterations": iterations,
        "pending_iteration": pending,
        "final_solution": final,
        "best_valid_solution": _best_system(
            initializer,
            iterations,
        ),
        "best_critic_candidate_oracle": _best_oracle(iterations),
        "errors": errors,
        "run_summary": {
            **summarize_api_calls(calls),
            "manifest_and_input_loading_seconds": loading_seconds,
            "accumulated_completed_iteration_wall_seconds": sum(
                float(
                    item.get("timing", {}).get(
                        "iteration_processing_wall_seconds",
                        0.0,
                    )
                )
                for item in iterations
            ),
            "current_invocation_wall_seconds_before_result_write": (
                invocation_seconds
            ),
        },
    }


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    run_id: str,
    model: str,
    candidate_count: int,
    candidate_strategy: str,
    fingerprint: str,
) -> None:
    expected = {
        "run_id": run_id,
        "model": model,
        "candidate_count_requested": candidate_count,
        "candidate_strategy": candidate_strategy,
        "problem_fingerprint_sha256": fingerprint,
    }
    actual = {
        key: checkpoint.get(key)
        for key in expected
    }
    if actual != expected:
        raise ValueError(
            "Checkpoint run-id/model/candidate-count/strateji/problem "
            "fingerprint ile uyuşmuyor."
        )


def _save_checkpoint(
    path: Path,
    *,
    run_id: str,
    model: str,
    candidate_count: int,
    candidate_strategy: str,
    fingerprint: str,
    initializer: dict[str, Any],
    iterations: list[dict[str, Any]],
    pending: dict[str, Any] | None,
    current_route: list[int],
    current_image: str,
    errors: list[dict[str, Any]],
) -> None:
    write_json(
        path,
        {
            "schema_version": "2.0",
            "run_id": run_id,
            "model": model,
            "candidate_count_requested": candidate_count,
            "candidate_strategy": candidate_strategy,
            "problem_fingerprint_sha256": fingerprint,
            "initializer": initializer,
            "iterations": iterations,
            "pending_iteration": pending,
            "current_route": current_route,
            "current_image": current_image,
            "errors": errors,
        },
    )


def _load_initializer(
    *,
    run_dir: Path,
    model_alias: str,
    problem: ProblemInstance,
    fingerprint: str,
) -> tuple[dict[str, Any], list[int], Path]:
    zero_path = (
        _model_root(run_dir, model_alias)
        / "zero_shot_results.json"
    )
    if not zero_path.exists():
        raise FileNotFoundError(
            "Önce aynı --run-id ve --model ile OpenRouter "
            "zero-shot deneyi çalıştırılmalıdır."
        )
    zero = read_json(zero_path)
    if (
        zero.get("problem", {}).get("fingerprint_sha256")
        != fingerprint
    ):
        raise ValueError(
            "OpenRouter zero-shot sonucu run manifestindeki "
            "problemle uyuşmuyor."
        )
    route = [
        int(value)
        for value in zero.get("route", [])
    ]
    evaluation = evaluate_route(problem, route)
    image_relative = (
        zero.get("artifacts", {}).get("route_image")
    )
    if not image_relative:
        raise ValueError(
            "Zero-shot sonucunda route_image artifactı bulunamadı."
        )
    image = _resolve_run_artifact(run_dir, image_relative)
    if not image.exists():
        raise FileNotFoundError(
            f"Zero-shot rota görseli bulunamadı: {image}"
        )
    initializer = _solution(
        "zero_shot",
        0,
        evaluation,
        source_file=_relative(zero_path, run_dir),
        route_image=image_relative,
    )
    return initializer, route, image


def _finish_scorer(
    pending: dict[str, Any],
    *,
    problem: ProblemInstance,
    model: str,
    output: Path,
    run_dir: Path,
    fallback_route: list[int],
    fallback_image: Path,
) -> tuple[dict[str, Any] | None, Exception | None]:
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
    ids = [
        candidate["candidate_id"]
        for candidate in candidates
    ]
    scores: dict[int, float] | None = None
    best_id: int | None = None
    raw_response: str | None = None
    scorer_call: dict[str, Any] | None = None
    reused_stored_response = False
    parsing_seconds = 0.0
    selection_mode = "visual_scorer_after_feasibility_filter"

    if not candidates:
        evaluation_timer = start_timer()
        evaluation = evaluate_route(problem, fallback_route)
        evaluation_seconds = elapsed_seconds(evaluation_timer)
        iteration = pending["iteration"]
        selected_image = (
            output
            / "images"
            / f"iteration_{iteration:02d}"
            / "selected.png"
        )
        rendering_timer = start_timer()
        plot_route(
            problem,
            fallback_route,
            selected_image,
            title=(
                f"{problem.name} Multi-Agent 1 — iteration "
                f"{iteration}: retained previous route"
            ),
        )
        rendering_seconds = elapsed_seconds(rendering_timer)
        scorer_seconds = elapsed_seconds(scorer_stage_timer)
        completed = {
            "iteration": iteration,
            "iteration_type": (
                "critic_candidates_then_"
                "feasibility_filtered_scorer"
            ),
            "critic": pending["critic"],
            "scorer": {
                "temperature": None,
                "selection_mode": (
                    "retain_previous_route_no_valid_candidate"
                ),
                "eligible_candidate_ids": [],
                "excluded_invalid_candidate_ids": excluded_ids,
                "raw_response": None,
                "scores": {},
                "best_candidate_id": None,
                "selection_regret_percent_after_evaluation": None,
                "api_call": None,
                "attempt_count": len(
                    pending.get("scorer_attempts", [])
                ),
                "timing": {
                    "reused_stored_response": False,
                    "api_call_wall_seconds": 0.0,
                    "response_parsing_seconds": 0.0,
                    "validation_and_metrics_seconds": evaluation_seconds,
                    "selected_route_rendering_seconds": rendering_seconds,
                    "scorer_stage_wall_seconds": scorer_seconds,
                },
            },
            "selected_solution": {
                **_evaluation_from(evaluation),
                "artifacts": {
                    "route_image": _relative(
                        selected_image,
                        run_dir,
                    ),
                    "retained_from_image": _relative(
                        fallback_image,
                        run_dir,
                    ),
                },
            },
            "timing": {
                "critic_stage_wall_seconds": pending[
                    "critic"
                ]["timing"]["critic_stage_wall_seconds"],
                "scorer_stage_wall_seconds": scorer_seconds,
                "critic_checkpoint_write_seconds": pending[
                    "critic"
                ]["timing"].get(
                    "checkpoint_write_seconds",
                    0.0,
                ),
                "completion_checkpoint_write_seconds": None,
                "checkpoint_write_seconds": pending[
                    "critic"
                ]["timing"].get(
                    "checkpoint_write_seconds",
                    0.0,
                ),
                "iteration_processing_wall_seconds": (
                    pending["critic"]["timing"][
                        "critic_stage_wall_seconds"
                    ]
                    + pending["critic"]["timing"].get(
                        "checkpoint_write_seconds",
                        0.0,
                    )
                    + scorer_seconds
                ),
            },
        }
        return completed, None

    if len(candidates) == 1:
        best_id = ids[0]
        scores = {}
        selection_mode = "single_valid_candidate_without_api"

    if best_id is None:
        for attempt in reversed(
            pending.get("scorer_attempts", [])
        ):
            try:
                parsing_timer = start_timer()
                scores, best_id = parse_scorer_response(
                    attempt["raw_response"],
                    expected_image_ids=ids,
                )
                parsing_seconds += elapsed_seconds(parsing_timer)
                raw_response = attempt["raw_response"]
                scorer_call = attempt.get("api_call")
                reused_stored_response = True
                break
            except Exception:
                parsing_seconds += elapsed_seconds(parsing_timer)

    if best_id is None:
        try:
            response = request_scorer(
                [
                    _resolve_run_artifact(
                        run_dir,
                        candidate["artifacts"]["route_image"],
                    )
                    for candidate in candidates
                ],
                problem=problem,
                image_ids=ids,
                model=model,
            )
            raw_response = response.text
            scorer_call = response.api_call
            pending.setdefault("scorer_attempts", []).append(
                {
                    "raw_response": raw_response,
                    "api_call": scorer_call,
                }
            )
            parsing_timer = start_timer()
            scores, best_id = parse_scorer_response(
                raw_response,
                expected_image_ids=ids,
            )
            parsing_seconds += elapsed_seconds(parsing_timer)
        except Exception as exc:
            if (
                scorer_call is not None
                and not hasattr(exc, "gemini_call_record")
            ):
                try:
                    setattr(exc, "gemini_call_record", scorer_call)
                except Exception:
                    pass
            try:
                setattr(
                    exc,
                    "ma1_scorer_stage_wall_seconds",
                    elapsed_seconds(scorer_stage_timer),
                )
            except Exception:
                pass
            return None, exc

    selected = next(
        item
        for item in candidates
        if item["candidate_id"] == best_id
    )
    valid = _valid(candidates)
    best_distance = min(
        item["distance"]
        for item in valid
    )
    selected_distance = selected.get("distance")
    regret = (
        100.0
        * (selected_distance - best_distance)
        / best_distance
        if selected_distance is not None and best_distance
        else None
    )
    iteration = pending["iteration"]
    selected_image = (
        output
        / "images"
        / f"iteration_{iteration:02d}"
        / "selected.png"
    )
    rendering_timer = start_timer()
    plot_route(
        problem,
        selected["route"],
        selected_image,
        title=(
            f"{problem.name} Multi-Agent 1 — iteration "
            f"{iteration}: selected candidate {best_id}"
        ),
    )
    rendering_seconds = elapsed_seconds(rendering_timer)
    scorer_seconds = elapsed_seconds(scorer_stage_timer)
    scorer_api_seconds = (
        float(
            scorer_call.get("api_call_wall_seconds", 0.0)
        )
        if scorer_call
        else 0.0
    )
    completed = {
        "iteration": iteration,
        "iteration_type": (
            "critic_candidates_then_"
            "feasibility_filtered_scorer"
        ),
        "critic": pending["critic"],
        "scorer": {
            "temperature": (
                0.0
                if selection_mode
                == "visual_scorer_after_feasibility_filter"
                else None
            ),
            "selection_mode": selection_mode,
            "eligible_candidate_ids": ids,
            "excluded_invalid_candidate_ids": excluded_ids,
            "raw_response": raw_response,
            "scores": {
                str(key): value
                for key, value in (scores or {}).items()
            },
            "best_candidate_id": best_id,
            "selection_regret_percent_after_evaluation": regret,
            "api_call": scorer_call,
            "attempt_count": len(
                pending.get("scorer_attempts", [])
            ),
            "timing": {
                "reused_stored_response": reused_stored_response,
                "api_call_wall_seconds": scorer_api_seconds,
                "response_parsing_seconds": parsing_seconds,
                "validation_and_metrics_seconds": 0.0,
                "selected_route_rendering_seconds": rendering_seconds,
                "scorer_stage_wall_seconds": scorer_seconds,
            },
        },
        "selected_solution": {
            **_evaluation_from(selected),
            "artifacts": {
                "route_image": _relative(
                    selected_image,
                    run_dir,
                ),
            },
        },
        "timing": {
            "critic_stage_wall_seconds": pending[
                "critic"
            ]["timing"]["critic_stage_wall_seconds"],
            "scorer_stage_wall_seconds": scorer_seconds,
            "critic_checkpoint_write_seconds": pending[
                "critic"
            ]["timing"].get(
                "checkpoint_write_seconds",
                0.0,
            ),
            "completion_checkpoint_write_seconds": None,
            "checkpoint_write_seconds": pending[
                "critic"
            ]["timing"].get(
                "checkpoint_write_seconds",
                0.0,
            ),
            "iteration_processing_wall_seconds": (
                pending["critic"]["timing"][
                    "critic_stage_wall_seconds"
                ]
                + pending["critic"]["timing"].get(
                    "checkpoint_write_seconds",
                    0.0,
                )
                + scorer_seconds
            ),
        },
    }
    return completed, None


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations en az 1 olmalıdır.")
    if not 1 <= args.candidate_count <= 7:
        raise SystemExit("--candidate-count 1 ile 7 arasında olmalıdır.")

    run_id = normalize_run_id(args.run_id)
    model = resolve_model_alias(args.model)
    run_dir = Path(args.output_dir) / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    output = (
        _model_root(run_dir, args.model)
        / "multi_agent1"
    )
    result_path = output / "multi_agent1_results.json"
    checkpoint_path = output / "multi_agent1_checkpoint.json"

    loading_timer = start_timer()
    if not manifest_path.exists():
        raise SystemExit(
            "Run manifesti bulunamadı. Önce baseline çalıştırılmalıdır."
        )
    manifest, problem = load_run_problem(manifest_path)
    fingerprint = manifest["problem"]["fingerprint_sha256"]
    try:
        initializer, current_route, current_image = _load_initializer(
            run_dir=run_dir,
            model_alias=args.model,
            problem=problem,
            fingerprint=fingerprint,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    loading_seconds = elapsed_seconds(loading_timer)

    if args.validate_only:
        print(
            "OpenRouter Multi-Agent 1 çevrimdışı doğrulaması "
            "başarılı."
        )
        print(f"Run ID: {run_id}")
        print(f"Problem: {problem.name}")
        print(f"Düğüm sayısı: {problem.dimension}")
        print(f"Depo: {problem.depot_id}")
        print(
            "Initializer geçerli: "
            f"{initializer['validation']['is_valid']}"
        )
        print(f"Hedef iterasyon: {args.iterations}")
        print(
            "Critic adayı / iterasyon: "
            f"{args.candidate_count}"
        )
        print(
            "Critic aday stratejisi: "
            f"{args.candidate_strategy}"
        )
        critic_requests = (
            args.candidate_count
            if args.candidate_strategy == "independent_calls"
            else 1
        )
        print(
            "Azami OpenRouter isteği: "
            f"{args.iterations * (critic_requests + 1)}"
        )
        print(
            "Tek ya da sıfır geçerli aday bulunan iterasyonlarda "
            "scorer isteği yapılmaz."
        )
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
        return

    invocation_timer = start_timer()
    iterations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    if args.resume and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        try:
            _validate_checkpoint(
                checkpoint,
                run_id=run_id,
                model=model,
                candidate_count=args.candidate_count,
                candidate_strategy=args.candidate_strategy,
                fingerprint=fingerprint,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        iterations = checkpoint.get("iterations", [])
        errors = checkpoint.get("errors", [])
        for error in errors:
            error_call = error.get("api_call")
            if isinstance(error_call, dict):
                error_call["success"] = False
        pending = checkpoint.get("pending_iteration")
        current_route = [
            int(value)
            for value in checkpoint["current_route"]
        ]
        current_image = _resolve_run_artifact(
            run_dir,
            checkpoint["current_image"],
        )
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
                f"\nİterasyon {pending['iteration']}: kayıtlı critic "
                "adayları yeniden kullanılacak; yalnız scorer "
                "aşaması yürütülecek."
            )
        else:
            print(
                f"\n=== Multi-Agent 1 iterasyon {next_iteration}: "
                f"{args.candidate_count} critic adayı ==="
            )
            critic_stage_timer = start_timer()
            try:
                request = request_candidates(
                    current_image,
                    problem=problem,
                    candidate_count=args.candidate_count,
                    model=model,
                    temperature=0.7,
                    strategy=args.candidate_strategy,
                )
                parsing_total = 0.0
                evaluation_total = 0.0
                rendering_total = 0.0
                candidates: list[dict[str, Any]] = []
                for candidate_id, raw in enumerate(
                    request.texts,
                    start=1,
                ):
                    candidate_timer = start_timer()

                    parsing_timer = start_timer()
                    parse_error: dict[str, str] | None = None
                    try:
                        route = parse_route(
                            raw,
                            depot_id=problem.depot_id,
                        )
                    except Exception as exc:
                        route = []
                        parse_error = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    parsing_seconds = elapsed_seconds(parsing_timer)
                    parsing_total += parsing_seconds

                    evaluation_timer = start_timer()
                    evaluation = evaluate_route(problem, route)
                    evaluation_seconds = elapsed_seconds(evaluation_timer)
                    evaluation_total += evaluation_seconds

                    image = (
                        output
                        / "images"
                        / f"iteration_{next_iteration:02d}"
                        / f"candidate_{candidate_id:02d}.png"
                    )
                    rendering_seconds = 0.0
                    image_relative: str | None = None
                    if (
                        evaluation["legal_node_ids"]
                        and len(route) >= 2
                    ):
                        rendering_timer = start_timer()
                        plot_route(
                            problem,
                            route,
                            image,
                            title=(
                                f"{problem.name} MA1 iteration "
                                f"{next_iteration} candidate "
                                f"{candidate_id}"
                            ),
                        )
                        rendering_seconds = elapsed_seconds(
                            rendering_timer
                        )
                        image_relative = _relative(image, run_dir)
                    rendering_total += rendering_seconds

                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "raw_response": raw,
                            "parse_error": parse_error,
                            **_evaluation_from(evaluation),
                            "artifacts": {
                                "route_image": image_relative,
                            },
                            "timing": {
                                "response_parsing_seconds": (
                                    parsing_seconds
                                ),
                                "validation_and_metrics_seconds": (
                                    evaluation_seconds
                                ),
                                "route_rendering_seconds": (
                                    rendering_seconds
                                ),
                                "candidate_processing_wall_seconds": (
                                    elapsed_seconds(candidate_timer)
                                ),
                            },
                        }
                    )
                    print(
                        f"Aday {candidate_id}: "
                        f"geçerli={evaluation['validation']['is_valid']}, "
                        f"mesafe={evaluation['distance']}, "
                        "referans gap="
                        f"{evaluation['gap_to_reference_percent']}"
                    )

                critic_seconds = elapsed_seconds(critic_stage_timer)
                pending = {
                    "iteration": next_iteration,
                    "critic": {
                        "temperature": 0.7,
                        "candidate_strategy": (
                            args.candidate_strategy
                        ),
                        "requested_candidate_count": (
                            args.candidate_count
                        ),
                        "returned_candidate_count": len(candidates),
                        "input_image": _relative(
                            current_image,
                            run_dir,
                        ),
                        "api_call": request.api_call,
                        "api_calls": request.api_calls,
                        "timing": {
                            "api_call_wall_seconds": request.api_call[
                                "api_call_wall_seconds"
                            ],
                            "response_parsing_seconds": parsing_total,
                            "validation_and_metrics_seconds": (
                                evaluation_total
                            ),
                            "route_rendering_seconds": rendering_total,
                            "critic_stage_wall_seconds": critic_seconds,
                        },
                        "candidates": candidates,
                    },
                    "scorer_attempts": [],
                }
                checkpoint_timer = start_timer()
                _save_checkpoint(
                    checkpoint_path,
                    run_id=run_id,
                    model=model,
                    candidate_count=args.candidate_count,
                    candidate_strategy=args.candidate_strategy,
                    fingerprint=fingerprint,
                    initializer=initializer,
                    iterations=iterations,
                    pending=pending,
                    current_route=current_route,
                    current_image=_relative(
                        current_image,
                        run_dir,
                    ),
                    errors=errors,
                )
                pending["critic"]["timing"][
                    "checkpoint_write_seconds"
                ] = elapsed_seconds(checkpoint_timer)
                _save_checkpoint(
                    checkpoint_path,
                    run_id=run_id,
                    model=model,
                    candidate_count=args.candidate_count,
                    candidate_strategy=args.candidate_strategy,
                    fingerprint=fingerprint,
                    initializer=initializer,
                    iterations=iterations,
                    pending=pending,
                    current_route=current_route,
                    current_image=_relative(
                        current_image,
                        run_dir,
                    ),
                    errors=errors,
                )
            except Exception as exc:
                failure_seconds = elapsed_seconds(critic_stage_timer)
                record = error_record(
                    exc,
                    phase="critic_candidate_generation",
                    iteration=next_iteration,
                )
                record["failed_stage_wall_seconds"] = failure_seconds
                errors.append(record)
                print(
                    f"Critic iterasyon {next_iteration} "
                    f"tamamlanamadı: {exc}"
                )
                stopped = True
                break

        assert pending is not None
        completed, scorer_error = _finish_scorer(
            pending,
            problem=problem,
            model=model,
            output=output,
            run_dir=run_dir,
            fallback_route=current_route,
            fallback_image=current_image,
        )
        if scorer_error is not None:
            record = error_record(
                scorer_error,
                phase="visual_scorer",
                iteration=pending["iteration"],
            )
            record["failed_stage_wall_seconds"] = getattr(
                scorer_error,
                "ma1_scorer_stage_wall_seconds",
                None,
            )
            errors.append(record)
            _save_checkpoint(
                checkpoint_path,
                run_id=run_id,
                model=model,
                candidate_count=args.candidate_count,
                candidate_strategy=args.candidate_strategy,
                fingerprint=fingerprint,
                initializer=initializer,
                iterations=iterations,
                pending=pending,
                current_route=current_route,
                current_image=_relative(current_image, run_dir),
                errors=errors,
            )
            print(
                f"Scorer iterasyon {pending['iteration']} "
                f"tamamlanamadı: {scorer_error}"
            )
            stopped = True
            break

        assert completed is not None
        selected = completed["selected_solution"]
        current_route = [
            int(value)
            for value in selected["route"]
        ]
        current_image = _resolve_run_artifact(
            run_dir,
            selected["artifacts"]["route_image"],
        )
        best_id = completed["scorer"]["best_candidate_id"]
        if best_id is None:
            print("Geçerli critic adayı yok; önceki rota korundu.")
        elif (
            completed["scorer"]["selection_mode"]
            == "single_valid_candidate_without_api"
        ):
            print(
                f"Tek geçerli aday otomatik seçildi: aday {best_id}"
            )
        else:
            print(f"Scorer seçimi: aday {best_id}")
        print(f"Seçilen mesafe: {selected['distance']}")

        iterations.append(completed)
        pending = None
        next_iteration = len(iterations) + 1
        checkpoint_timer = start_timer()
        _save_checkpoint(
            checkpoint_path,
            run_id=run_id,
            model=model,
            candidate_count=args.candidate_count,
            candidate_strategy=args.candidate_strategy,
            fingerprint=fingerprint,
            initializer=initializer,
            iterations=iterations,
            pending=None,
            current_route=current_route,
            current_image=_relative(current_image, run_dir),
            errors=errors,
        )
        checkpoint_seconds = elapsed_seconds(checkpoint_timer)
        completed["timing"][
            "completion_checkpoint_write_seconds"
        ] = checkpoint_seconds
        completed["timing"]["checkpoint_write_seconds"] = (
            float(
                completed["timing"].get(
                    "critic_checkpoint_write_seconds",
                    0.0,
                )
            )
            + checkpoint_seconds
        )
        completed["timing"][
            "iteration_processing_wall_seconds"
        ] += checkpoint_seconds
        _save_checkpoint(
            checkpoint_path,
            run_id=run_id,
            model=model,
            candidate_count=args.candidate_count,
            candidate_strategy=args.candidate_strategy,
            fingerprint=fingerprint,
            initializer=initializer,
            iterations=iterations,
            pending=None,
            current_route=current_route,
            current_image=_relative(current_image, run_dir),
            errors=errors,
        )

    result = _result(
        run_id=run_id,
        problem=problem,
        fingerprint=fingerprint,
        model_alias=args.model,
        model=model,
        candidate_count=args.candidate_count,
        candidate_strategy=args.candidate_strategy,
        requested=args.iterations,
        initializer=initializer,
        iterations=iterations,
        pending=pending,
        errors=errors,
        loading_seconds=loading_seconds,
        invocation_seconds=elapsed_seconds(invocation_timer),
    )
    write_json(result_path, result)
    print(
        "\nOpenRouter dinamik Multi-Agent 1 durumu kaydedildi."
    )
    print(f"Model: {args.model}")
    print(f"Tamamlanan tam iterasyon: {len(iterations)}")
    if pending:
        print(
            f"Scorer bekleyen iterasyon: {pending['iteration']} "
            "(--resume ile devam edilir)"
        )
    print(f"Sonuç dosyası: {result_path}")


if __name__ == "__main__":
    main()
