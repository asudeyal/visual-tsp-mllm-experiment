"""OpenRouter modelleri için initializer + iteratif critic Multi-Agent 2."""

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
from src.gemini import critic_prompt, parse_route
from src.metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
)
from src.openrouter import (
    OPENROUTER_MODELS,
    request_route,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Girdileri API çağrısı yapmadan doğrular.",
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


def _result(
    *,
    run_id: str,
    problem: ProblemInstance,
    fingerprint: str,
    model_alias: str,
    model: str,
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
        "experiment": "dynamic_openrouter_multi_agent_2_tsp",
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
            "provider": "openrouter",
            "alias": model_alias,
            "requested_name": model,
            "critic_temperature": 0.7,
            "reasoning_effort": "none",
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
    model = resolve_model_alias(args.model)
    run_dir = Path(args.output_dir) / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    output = (
        _model_root(run_dir, args.model)
        / "multi_agent2"
    )
    result_path = output / "multi_agent2_results.json"
    checkpoint_path = output / "multi_agent2_checkpoint.json"

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

    prompt_timer = start_timer()
    prompt = critic_prompt(problem)
    prompt_seconds = elapsed_seconds(prompt_timer)

    if args.validate_only:
        print(
            "OpenRouter Multi-Agent 2 çevrimdışı doğrulaması "
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
            "Tahmini kalan OpenRouter isteği: "
            f"{args.iterations}"
        )
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
        return

    invocation_timer = start_timer()
    iterations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

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

    for iteration_number in range(
        len(iterations) + 1,
        args.iterations + 1,
    ):
        iteration_timer = start_timer()
        print(
            f"\n--- Critic iterasyon {iteration_number} ---"
        )
        current_phase = "prompt_preparation"
        iteration_prompt_timer = start_timer()
        prompt = critic_prompt(problem)
        iteration_prompt_seconds = elapsed_seconds(
            iteration_prompt_timer
        )
        try:
            current_phase = "critic_route_revision"
            request = request_route(
                current_image,
                prompt=prompt,
                model=model,
                temperature=0.7,
                max_tokens=8192,
                reasoning_effort="none",
                phase="critic_route_revision",
            )

            current_phase = "response_parsing"
            parsing_timer = start_timer()
            route = parse_route(
                request.text,
                depot_id=problem.depot_id,
            )
            parsing_seconds = elapsed_seconds(parsing_timer)

            current_phase = "validation_and_metrics"
            evaluation_timer = start_timer()
            evaluation = evaluate_route(problem, route)
            evaluation_seconds = elapsed_seconds(evaluation_timer)

            current_phase = "route_rendering"
            rendering_timer = start_timer()
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
                **_evaluation_from(evaluation),
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
                    "response_parsing_seconds": parsing_seconds,
                    "validation_and_metrics_seconds": (
                        evaluation_seconds
                    ),
                    "route_rendering_seconds": rendering_seconds,
                    "checkpoint_write_seconds": None,
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
            record["timing"]["iteration_total_wall_seconds"] = (
                elapsed_seconds(iteration_timer)
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
            record["timing"]["iteration_total_wall_seconds"] = (
                elapsed_seconds(iteration_timer)
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
                "İterasyon toplam süresi: "
                f"{record['timing']['iteration_total_wall_seconds']:.4f} sn"
            )
        except Exception as exc:
            failure = error_record(
                exc,
                phase=current_phase,
                iteration=iteration_number,
            )
            previous_attempts = sum(
                error.get("iteration") == iteration_number
                for error in errors
            )
            failure["attempt"] = previous_attempts + 1
            failure["timing"] = {
                "prompt_preparation_seconds": (
                    iteration_prompt_seconds
                ),
                "failed_attempt_wall_seconds": elapsed_seconds(
                    iteration_timer
                ),
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
            break

    result = _result(
        run_id=run_id,
        problem=problem,
        fingerprint=fingerprint,
        model_alias=args.model,
        model=model,
        requested=args.iterations,
        initializer=initializer,
        iterations=iterations,
        errors=errors,
        input_loading_seconds=loading_seconds,
        invocation_seconds=elapsed_seconds(invocation_timer),
    )
    write_json(result_path, result)

    print(
        "\nOpenRouter dinamik Multi-Agent 2 durumu kaydedildi."
    )
    print(f"Model: {args.model}")
    print(f"Problem: {problem.name}")
    print(f"Tamamlanan iterasyon: {len(iterations)}")
    if len(iterations) < args.iterations:
        print(
            "Kalan iterasyonlar --resume ile devam ettirilebilir."
        )
    print(f"Sonuç dosyası: {result_path}")


if __name__ == "__main__":
    main()
