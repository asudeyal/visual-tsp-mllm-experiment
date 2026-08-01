"""Aynı Multi-Agent 2 akışını Gemini, OpenRouter veya Groq ile çalıştırır."""

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
from src.experiment_observability import (
    ExperimentObservability,
    add_observability_arguments,
    settings_from_args,
)
from src.gemini import critic_prompt, parse_route
from src.metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
)
from src.problem_instance import ProblemInstance
from src.providers import (
    ProviderAdapter,
    create_provider,
    provider_model_root,
    supported_providers,
    zero_shot_result_candidates,
)
from src.run_manifest import load_run_problem
from src.solution_tracking import SolutionProgressTracker


ROOT = Path(__file__).resolve().parent
EVALUATION_KEYS = (
    "route",
    "validation",
    "legal_node_ids",
    "distance",
    "reference_distance",
    "gap_to_reference_percent",
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
        "--provider",
        required=True,
        choices=supported_providers(),
    )
    parser.add_argument(
        "--model",
        help="Model adı veya OpenRouter kısa model adı.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Girdileri API çağrısı yapmadan doğrular.",
    )
    add_observability_arguments(
        parser,
        include_early_stop=True,
    )
    return parser.parse_args()


def _evaluation_from(
    value: dict[str, Any],
) -> dict[str, Any]:
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


def _best(
    initializer: dict[str, Any],
    iterations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    choices = [initializer] + [
        _solution(
            "critic",
            item["iteration"],
            item,
        )
        for item in iterations
    ]
    valid = [
        item
        for item in choices
        if (
            item.get("validation", {}).get("is_valid") is True
            and item.get("distance") is not None
        )
    ]
    return (
        min(valid, key=lambda item: item["distance"])
        if valid
        else None
    )


def _relative(
    path: Path,
    run_dir: Path,
) -> str:
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


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    run_id: str,
    model: str,
    fingerprint: str,
) -> None:
    expected = {
        "run_id": run_id,
        "model": model,
        "problem_fingerprint_sha256": fingerprint,
    }
    actual = {
        key: checkpoint.get(key)
        for key in expected
    }
    if actual != expected:
        raise ValueError(
            "Checkpoint run-id/model/problem fingerprint ile uyuşmuyor."
        )


def _save_checkpoint(
    path: Path,
    *,
    run_id: str,
    model: str,
    fingerprint: str,
    initializer: dict[str, Any],
    iterations: list[dict[str, Any]],
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
            "problem_fingerprint_sha256": fingerprint,
            "initializer": initializer,
            "iterations": iterations,
            "current_route": current_route,
            "current_image": current_image,
            "errors": errors,
        },
    )


def _deduplicated_calls(
    iterations: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls = [
        item["api_call"]
        for item in iterations
        if isinstance(item.get("api_call"), dict)
    ]

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
        call = error.get("api_call")
        if isinstance(call, dict) and identity(call) not in known:
            calls.append(call)
            known.add(identity(call))
    return calls


def _request_timing(
    api_call: dict[str, Any] | None,
    *,
    request_control: dict[str, Any] | None = None,
) -> dict[str, float]:
    call = api_call if isinstance(api_call, dict) else {}
    api_wall = float(
        call.get("api_call_wall_seconds") or 0.0
    )
    control = request_control
    if not isinstance(control, dict):
        candidate = call.get("request_control")
        control = candidate if isinstance(candidate, dict) else None

    if not isinstance(control, dict):
        return {
            "api_active_wall_seconds": api_wall,
            "deliberate_delay_seconds": 0.0,
            "rate_limit_backoff_seconds": 0.0,
            "controlled_wait_seconds": 0.0,
            "api_request_total_wall_seconds": api_wall,
        }

    waits = control.get("waits")
    if not isinstance(waits, dict):
        waits = {}

    deliberate = float(
        waits.get("deliberate_delay_seconds") or 0.0
    )
    backoff = float(
        waits.get("rate_limit_backoff_seconds") or 0.0
    )
    controlled = float(
        waits.get("controlled_wait_seconds")
        or deliberate + backoff
    )

    return {
        "api_active_wall_seconds": float(
            control.get("active_wall_seconds") or api_wall
        ),
        "deliberate_delay_seconds": deliberate,
        "rate_limit_backoff_seconds": backoff,
        "controlled_wait_seconds": controlled,
        "api_request_total_wall_seconds": float(
            control.get("total_wall_seconds")
            or api_wall + controlled
        ),
    }


def _token_count(
    api_call: dict[str, Any] | None,
) -> int | None:
    if not isinstance(api_call, dict):
        return None
    usage = api_call.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("total_token_count")
    return int(value) if value is not None else None


def _create_progress_tracker(
    *,
    problem: ProblemInstance,
    provider: ProviderAdapter,
    observability: ExperimentObservability,
) -> SolutionProgressTracker:
    reference = problem.reference
    return SolutionProgressTracker(
        provider=provider.provider_id,
        reference_distance=(
            reference.distance
            if reference is not None
            else None
        ),
        reference_type=(
            reference.reference_type.value
            if reference is not None
            else None
        ),
        reference_is_proven_optimal=(
            reference.is_proven_optimal
            if reference is not None
            else False
        ),
        early_stop_policy=observability.early_stop_policy(),
    )


def _replay_solution_progress(
    tracker: SolutionProgressTracker,
    *,
    initializer: dict[str, Any],
    iterations: list[dict[str, Any]],
) -> None:
    tracker.seed_initializer(initializer)
    for item in iterations:
        progress = tracker.record_multi_agent2_iteration(
            iteration=int(item["iteration"]),
            solution=item,
        )
        item["solution_progress"] = progress
        gbest = progress.get("system_gbest") or {}
        item["iteration_best_distance"] = (
            (progress.get("iteration_best") or {}).get("distance")
        )
        item["system_gbest_distance"] = gbest.get("distance")
        item["system_gbest_gap_percent"] = gbest.get(
            "gap_to_reference_percent"
        )


def _result(
    *,
    run_id: str,
    problem: ProblemInstance,
    fingerprint: str,
    provider: ProviderAdapter,
    requested: int,
    initializer: dict[str, Any],
    iterations: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    input_loading_seconds: float,
    invocation_seconds: float,
) -> dict[str, Any]:
    final = (
        _solution(
            "critic",
            iterations[-1]["iteration"],
            iterations[-1],
        )
        if iterations
        else initializer
    )
    calls = _deduplicated_calls(iterations, errors)
    return {
        "schema_version": "2.0",
        "experiment": "dynamic_unified_multi_agent_2_tsp",
        "run_id": run_id,
        "method": "multi_agent_2",
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
            **provider.model_metadata,
            "critic_temperature": 0.7,
        },
        "requested_iterations": requested,
        "completed_iterations": len(iterations),
        "artificial_delay_enabled": False,
        "initializer": initializer,
        "iterations": iterations,
        "final_solution": final,
        "best_valid_solution": _best(
            initializer,
            iterations,
        ),
        "errors": errors,
        "run_summary": {
            **summarize_api_calls(calls),
            "manifest_and_input_loading_seconds": (
                input_loading_seconds
            ),
            "accumulated_completed_iteration_wall_seconds": sum(
                float(
                    item.get("timing", {}).get(
                        "iteration_total_wall_seconds",
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


def _load_initializer(
    *,
    run_dir: Path,
    provider: ProviderAdapter,
    problem: ProblemInstance,
    fingerprint: str,
) -> tuple[dict[str, Any], list[int], Path]:
    candidates = zero_shot_result_candidates(
        run_dir,
        provider.provider_id,
        provider.model_alias,
    )
    zero_path = next(
        (path for path in candidates if path.exists()),
        None,
    )
    if zero_path is None:
        raise FileNotFoundError(
            "Önce aynı --run-id, --provider ve --model ile ortak "
            "run_zero_shot.py çalıştırılmalıdır. Uyumlu tarihsel "
            "zero-shot sonucu da bulunamadı."
        )
    zero = read_json(zero_path)
    if (
        zero.get("problem", {}).get("fingerprint_sha256")
        != fingerprint
    ):
        raise ValueError(
            "Zero-shot sonucu run manifestindeki "
            "problemle uyuşmuyor."
        )
    model_value = zero.get("model")
    if isinstance(model_value, str):
        model_names = {model_value}
    else:
        model_names = {
            str(value)
            for key in (
                "alias",
                "name",
                "requested_name",
                "response_name",
            )
            if (
                value := (model_value or {}).get(key)
            )
        }
    if not {
        provider.model_alias,
        provider.resolved_model,
    }.intersection(model_names):
        raise ValueError(
            "Zero-shot sonucu seçilen provider/model ile uyuşmuyor."
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
    image = _resolve_run_artifact(
        run_dir,
        image_relative,
    )
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


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations en az 1 olmalıdır.")

    run_id = normalize_run_id(args.run_id)
    try:
        provider = create_provider(args.provider, args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    settings = settings_from_args(
        args,
        include_early_stop=True,
    )
    observability = ExperimentObservability(settings)
    if not args.validate_only:
        provider.configure_request_controller(
            observability.request_controller
        )
        observability.start()

    model = provider.resolved_model
    run_dir = Path(args.output_dir) / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    output = (
        provider_model_root(
            run_dir,
            provider.provider_id,
            provider.model_alias,
        )
        / "multi_agent2"
    )
    result_path = output / "multi_agent2_results.json"
    checkpoint_path = output / "multi_agent2_checkpoint.json"

    loading_timer = start_timer()
    with observability.phase("manifest_and_input_loading"):
        if not manifest_path.exists():
            observability.stop()
            raise SystemExit(
                "Run manifesti bulunamadı. Önce baseline "
                "çalıştırılmalıdır."
            )
        manifest, problem = load_run_problem(manifest_path)
        fingerprint = manifest["problem"]["fingerprint_sha256"]
        try:
            initializer, current_route, current_image = (
                _load_initializer(
                    run_dir=run_dir,
                    provider=provider,
                    problem=problem,
                    fingerprint=fingerprint,
                )
            )
        except Exception as exc:
            observability.stop()
            raise SystemExit(str(exc)) from exc
    loading_seconds = elapsed_seconds(loading_timer)

    prompt_timer = start_timer()
    with observability.phase("prompt_preparation"):
        prompt = critic_prompt(problem)
    prompt_seconds = elapsed_seconds(prompt_timer)

    if args.validate_only:
        print(
            "Birleşik Multi-Agent 2 çevrimdışı doğrulaması "
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
        print(f"Critic prompt karakter sayısı: {len(prompt)}")
        print(
            f"Tahmini kalan {provider.provider_id} isteği: "
            f"{args.iterations}"
        )
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
        return

    invocation_timer = start_timer()
    iterations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    termination: dict[str, Any] = {
        "reason": "requested_iterations_completed",
        "early_stop": None,
        "failed_iteration": None,
    }

    if args.resume and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        try:
            _validate_checkpoint(
                checkpoint,
                run_id=run_id,
                model=model,
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
        current_route = [
            int(value)
            for value in checkpoint["current_route"]
        ]
        current_image = _resolve_run_artifact(
            run_dir,
            checkpoint["current_image"],
        )
        print(
            f"Checkpoint yüklendi: {len(iterations)} "
            "iterasyon tamamlanmış. "
            f"İterasyon {len(iterations) + 1}'den devam ediliyor."
        )
    elif args.resume:
        print(
            "Checkpoint bulunamadı; deney baştan başlatılıyor."
        )

    progress_tracker = _create_progress_tracker(
        problem=problem,
        provider=provider,
        observability=observability,
    )
    _replay_solution_progress(
        progress_tracker,
        initializer=initializer,
        iterations=iterations,
    )
    if progress_tracker.should_stop:
        termination = {
            "reason": "early_stop",
            "early_stop": progress_tracker.latest_early_stop,
            "failed_iteration": None,
        }

    for iteration_number in range(
        len(iterations) + 1,
        args.iterations + 1,
    ):
        if progress_tracker.should_stop:
            break

        iteration_timer = start_timer()
        request = None
        print(
            f"\n--- Critic iterasyon {iteration_number} ---"
        )
        current_phase = "prompt_preparation"
        iteration_prompt_timer = start_timer()
        with observability.phase("prompt_preparation"):
            prompt = critic_prompt(problem)
        iteration_prompt_seconds = elapsed_seconds(
            iteration_prompt_timer
        )
        try:
            current_phase = "critic_route_revision"
            with observability.phase("api_request"):
                request = provider.request_route(
                    current_image,
                    prompt=prompt,
                    temperature=0.7,
                    phase="critic_route_revision",
                )

            current_phase = "response_parsing"
            parsing_timer = start_timer()
            with observability.phase("response_parsing"):
                route = parse_route(
                    request.text,
                    depot_id=problem.depot_id,
                )
            parsing_seconds = elapsed_seconds(parsing_timer)

            current_phase = "validation_and_metrics"
            evaluation_timer = start_timer()
            with observability.phase("validation_and_metrics"):
                evaluation = evaluate_route(problem, route)
            evaluation_seconds = elapsed_seconds(evaluation_timer)

            current_phase = "route_rendering"
            rendering_timer = start_timer()
            with observability.phase("route_rendering"):
                image_path = (
                    output
                    / "images"
                    / f"iteration_{iteration_number:02d}.png"
                )
                plot_route(
                    problem,
                    route,
                    image_path,
                    title=(
                        f"{problem.name} Multi-Agent 2 — "
                        f"iteration {iteration_number}"
                    ),
                )
            rendering_seconds = elapsed_seconds(rendering_timer)

            request_timing = _request_timing(request.api_call)
            token_count = _token_count(request.api_call)
            progress = progress_tracker.record_multi_agent2_iteration(
                iteration=iteration_number,
                solution=evaluation,
            )
            iteration_best = progress.get("iteration_best") or {}
            system_gbest = progress.get("system_gbest") or {}

            record = {
                "iteration": iteration_number,
                "iteration_type": "critic_route_revision",
                "status": "completed",
                "temperature": 0.7,
                "input_image": _relative(
                    current_image,
                    run_dir,
                ),
                "raw_response": request.text,
                "api_call": request.api_call,
                "token_count": token_count,
                **_evaluation_from(evaluation),
                "solution_progress": progress,
                "iteration_best_distance": iteration_best.get(
                    "distance"
                ),
                "system_gbest_distance": system_gbest.get(
                    "distance"
                ),
                "system_gbest_gap_percent": system_gbest.get(
                    "gap_to_reference_percent"
                ),
                "artifacts": {
                    "route_image": _relative(
                        image_path,
                        run_dir,
                    ),
                },
                "timing": {
                    "prompt_preparation_seconds": (
                        iteration_prompt_seconds
                    ),
                    "api_call_wall_seconds": request.api_call[
                        "api_call_wall_seconds"
                    ],
                    **request_timing,
                    "response_parsing_seconds": parsing_seconds,
                    "validation_and_metrics_seconds": (
                        evaluation_seconds
                    ),
                    "route_rendering_seconds": rendering_seconds,
                    "checkpoint_write_seconds": None,
                    "iteration_active_wall_seconds": None,
                    "iteration_total_wall_seconds": None,
                },
            }
            iterations.append(record)
            current_route = route
            current_image = image_path

            current_phase = "checkpoint_write"
            checkpoint_timer = start_timer()
            _save_checkpoint(
                checkpoint_path,
                run_id=run_id,
                model=model,
                fingerprint=fingerprint,
                initializer=initializer,
                iterations=iterations,
                current_route=current_route,
                current_image=_relative(
                    current_image,
                    run_dir,
                ),
                errors=errors,
            )
            checkpoint_seconds = elapsed_seconds(
                checkpoint_timer
            )
            record["timing"]["checkpoint_write_seconds"] = (
                checkpoint_seconds
            )
            iteration_total = elapsed_seconds(iteration_timer)
            record["timing"]["iteration_total_wall_seconds"] = (
                iteration_total
            )
            record["timing"]["iteration_active_wall_seconds"] = max(
                0.0,
                iteration_total
                - request_timing["controlled_wait_seconds"],
            )
            # İlk yazımın süresini checkpoint'e de kaydetmek için
            # güncellenmiş kayıt bir kez daha yazılır.
            _save_checkpoint(
                checkpoint_path,
                run_id=run_id,
                model=model,
                fingerprint=fingerprint,
                initializer=initializer,
                iterations=iterations,
                current_route=current_route,
                current_image=_relative(
                    current_image,
                    run_dir,
                ),
                errors=errors,
            )
            iteration_total = elapsed_seconds(iteration_timer)
            record["timing"]["iteration_total_wall_seconds"] = (
                iteration_total
            )
            record["timing"]["iteration_active_wall_seconds"] = max(
                0.0,
                iteration_total
                - request_timing["controlled_wait_seconds"],
            )

            print(
                "Geçerli mi? "
                f"{evaluation['validation']['is_valid']}"
            )
            print(f"Mesafe: {evaluation['distance']}")
            gap = evaluation["gap_to_reference_percent"]
            print(
                "Referans gap: "
                f"{'hesaplanamadı' if gap is None else f'%{gap:.4f}'}"
            )
            print(
                "İterasyon en iyi / sistem GBest: "
                f"{record['iteration_best_distance']} / "
                f"{record['system_gbest_distance']}"
            )
            print(
                "Token / aktif / bekleme / toplam: "
                f"{token_count if token_count is not None else '-'} / "
                f"{record['timing']['iteration_active_wall_seconds']:.4f} / "
                f"{request_timing['controlled_wait_seconds']:.4f} / "
                f"{record['timing']['iteration_total_wall_seconds']:.4f} sn"
            )

            if progress_tracker.should_stop:
                termination = {
                    "reason": "early_stop",
                    "early_stop": (
                        progress_tracker.latest_early_stop
                    ),
                    "failed_iteration": None,
                }
                print(
                    "Erken durdurma: sistem GBest gap değeri "
                    f"%{settings.early_stop_gap_percent:g} "
                    "eşiğine ulaştı."
                )
                break
        except Exception as exc:
            failure = error_record(
                exc,
                phase=current_phase,
                iteration=iteration_number,
            )
            if request is not None:
                failure["raw_response"] = request.text
                failure.setdefault("api_call", request.api_call)
            request_control = getattr(
                exc,
                "request_control_report",
                None,
            )
            if isinstance(request_control, dict):
                failure["request_control"] = request_control
            previous_attempts = sum(
                error.get("iteration") == iteration_number
                for error in errors
            )
            failure["attempt"] = previous_attempts + 1
            failure_request_timing = _request_timing(
                failure.get("api_call"),
                request_control=(
                    request_control
                    if isinstance(request_control, dict)
                    else None
                ),
            )
            failed_total = elapsed_seconds(iteration_timer)
            failure["timing"] = {
                "prompt_preparation_seconds": (
                    iteration_prompt_seconds
                ),
                **failure_request_timing,
                "failed_attempt_active_wall_seconds": max(
                    0.0,
                    failed_total
                    - failure_request_timing[
                        "controlled_wait_seconds"
                    ],
                ),
                "failed_attempt_wall_seconds": failed_total,
            }
            errors.append(failure)
            _save_checkpoint(
                checkpoint_path,
                run_id=run_id,
                model=model,
                fingerprint=fingerprint,
                initializer=initializer,
                iterations=iterations,
                current_route=current_route,
                current_image=_relative(
                    current_image,
                    run_dir,
                ),
                errors=errors,
            )
            print(
                f"Critic iterasyon {iteration_number} "
                f"tamamlanamadı: {exc}"
            )
            termination = {
                "reason": "iteration_failed",
                "early_stop": (
                    progress_tracker.latest_early_stop
                ),
                "failed_iteration": iteration_number,
            }
            break

    try:
        observability_summary = observability.stop()
        result = _result(
            run_id=run_id,
            problem=problem,
            fingerprint=fingerprint,
            provider=provider,
            requested=args.iterations,
            initializer=initializer,
            iterations=iterations,
            errors=errors,
            input_loading_seconds=loading_seconds,
            invocation_seconds=elapsed_seconds(invocation_timer),
        )
        result["artificial_delay_enabled"] = (
            settings.minimum_request_interval_seconds > 0
        )
        result["solution_progress"] = progress_tracker.snapshot()
        result["termination"] = termination
        result["observability"] = observability_summary
        result["run_summary"].update(
            {
                "accumulated_completed_iteration_active_wall_seconds": sum(
                    float(
                        item.get("timing", {}).get(
                            "iteration_active_wall_seconds",
                            0.0,
                        )
                    )
                    for item in iterations
                ),
                "accumulated_deliberate_delay_seconds": sum(
                    float(
                        item.get("timing", {}).get(
                            "deliberate_delay_seconds",
                            0.0,
                        )
                    )
                    for item in iterations
                ),
                "accumulated_rate_limit_backoff_seconds": sum(
                    float(
                        item.get("timing", {}).get(
                            "rate_limit_backoff_seconds",
                            0.0,
                        )
                    )
                    for item in iterations
                ),
                "accumulated_controlled_wait_seconds": sum(
                    float(
                        item.get("timing", {}).get(
                            "controlled_wait_seconds",
                            0.0,
                        )
                    )
                    for item in iterations
                ),
            }
        )
        write_json(result_path, result)
    finally:
        observability.stop()

    print(
        "\nBirleşik dinamik Multi-Agent 2 durumu kaydedildi."
    )
    print(f"Provider: {provider.provider_id}")
    print(f"Model: {provider.model_alias}")
    print(f"Problem: {problem.name}")
    print(f"Tamamlanan iterasyon: {len(iterations)}")
    if termination["reason"] == "early_stop":
        early_stop = termination["early_stop"] or {}
        print(
            "Durdurma: erken durdurma, sistem GBest gap="
            f"{early_stop.get('system_gbest_gap_percent')}%"
        )
    elif len(iterations) < args.iterations:
        print(
            "Kalan iterasyonlar --resume ile devam ettirilebilir."
        )
    request_summary = result["observability"]["request_control"]
    waits = request_summary["waits"]
    print(
        "API denemesi/retry: "
        f"{request_summary['request_attempt_count']}/"
        f"{request_summary['retry_count']}"
    )
    print(
        "Bilinçli bekleme / rate-limit backoff: "
        f"{waits['deliberate_delay_seconds']:.4f} / "
        f"{waits['rate_limit_backoff_seconds']:.4f} sn"
    )
    print(f"Sonuç dosyası: {result_path}")


if __name__ == "__main__":
    main()
