"""Aynı zero-shot TSP akışını Gemini, OpenRouter veya Groq ile çalıştırır."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core import (
    evaluate_route,
    normalize_run_id,
    plot_route,
    write_json,
)
from src.experiment_observability import (
    ExperimentObservability,
    add_observability_arguments,
    settings_from_args,
)
from src.gemini import (
    initializer_prompt,
    parse_route,
)
from src.metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
)
from src.providers import (
    create_provider,
    provider_model_root,
    supported_providers,
)
from src.run_manifest import load_run_problem


ROOT = Path(__file__).resolve().parent


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
        help=(
            "Model adı veya OpenRouter kısa adı. Gemini/Groq için "
            "verilmezse kayıtlı varsayılan model kullanılır."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Manifest ve promptu API çağrısı yapmadan doğrular.",
    )
    add_observability_arguments(
        parser,
        include_early_stop=False,
    )
    return parser.parse_args()


def _relative(
    path: Path,
    run_dir: Path,
) -> str:
    return path.resolve().relative_to(
        run_dir.resolve()
    ).as_posix()


def _request_timing(
    api_call: dict,
) -> dict[str, float]:
    api_wall = float(
        api_call.get("api_call_wall_seconds") or 0.0
    )
    request_control = api_call.get("request_control")
    if not isinstance(request_control, dict):
        return {
            "api_active_wall_seconds": api_wall,
            "controlled_wait_seconds": 0.0,
            "api_request_total_wall_seconds": api_wall,
        }

    waits = request_control.get("waits")
    if not isinstance(waits, dict):
        waits = {}

    return {
        "api_active_wall_seconds": float(
            request_control.get("active_wall_seconds")
            or api_wall
        ),
        "controlled_wait_seconds": float(
            waits.get("controlled_wait_seconds") or 0.0
        ),
        "api_request_total_wall_seconds": float(
            request_control.get("total_wall_seconds")
            or api_wall
        ),
    }


def main() -> None:
    args = parse_args()
    run_id = normalize_run_id(args.run_id)

    try:
        provider = create_provider(args.provider, args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    settings = settings_from_args(
        args,
        include_early_stop=False,
    )
    observability = ExperimentObservability(settings)

    if not args.validate_only:
        provider.configure_request_controller(
            observability.request_controller
        )
        observability.start()

    run_dir = Path(args.output_dir) / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    output = (
        provider_model_root(
            run_dir,
            provider.provider_id,
            provider.model_alias,
        )
        / "zero_shot"
    )
    result_path = output / "zero_shot_results.json"
    total_timer = start_timer()

    loading_timer = start_timer()
    with observability.phase("manifest_and_input_loading"):
        if not manifest_path.exists():
            observability.stop()
            raise SystemExit(
                "Run manifesti bulunamadı. Önce aynı --run-id ile "
                "run_baseline.py çalıştırılmalıdır."
            )
        manifest, problem = load_run_problem(manifest_path)
        points_image = run_dir / "baseline" / "images" / "points.png"
        if not points_image.exists():
            observability.stop()
            raise SystemExit(
                f"Baseline problem görseli bulunamadı: {points_image}"
            )
    loading_seconds = elapsed_seconds(loading_timer)

    prompt_timer = start_timer()
    with observability.phase("prompt_preparation"):
        prompt = initializer_prompt(problem)
    prompt_seconds = elapsed_seconds(prompt_timer)

    if args.validate_only:
        print("Zero-shot çevrimdışı doğrulaması başarılı.")
        print(f"Run ID: {run_id}")
        print(f"Problem: {problem.name}")
        print(f"Düğüm sayısı: {problem.dimension}")
        print(f"Depo: {problem.depot_id}")
        print(f"Provider: {provider.provider_id}")
        print(f"Model: {provider.model_alias}")
        print(f"Çözümlenen model: {provider.resolved_model}")
        print(
            "Problem fingerprint: "
            f"{manifest['problem']['fingerprint_sha256']}"
        )
        print(f"Prompt karakter sayısı: {len(prompt)}")
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
        return

    calls: list[dict] = []
    errors: list[dict] = []
    result: dict = {
        "schema_version": "2.0",
        "experiment": "dynamic_unified_visual_zero_shot_tsp",
        "run_id": run_id,
        "method": "zero_shot",
        "problem": {
            "name": problem.name,
            "dimension": problem.dimension,
            "depot_id": problem.depot_id,
            "source_type": problem.source_type.value,
            "fingerprint_sha256": manifest["problem"][
                "fingerprint_sha256"
            ],
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
            "temperature": 0.0,
        },
        "model_input": {
            "image": _relative(points_image, run_dir),
            "coordinates_sent_to_model": False,
            "prompt": prompt,
        },
        "errors": errors,
    }

    current_phase = "route_generation"

    try:
        try:
            with observability.phase("api_request"):
                request = provider.request_route(
                    points_image,
                    prompt=prompt,
                    temperature=0.0,
                    phase="route_generation",
                )
            calls.append(request.api_call)
            result["raw_response"] = request.text

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
                image_path = output / "images" / "route.png"
                plot_route(
                    problem,
                    route,
                    image_path,
                    title=(
                        f"{provider.provider_id}/{provider.model_alias} "
                        f"zero-shot {problem.name} — "
                        f"distance {evaluation['distance']}"
                    ),
                )
            rendering_seconds = elapsed_seconds(rendering_timer)

            request_timing = _request_timing(request.api_call)
            result.update(
                {
                    **evaluation,
                    "artifacts": {
                        "route_image": _relative(
                            image_path,
                            run_dir,
                        ),
                    },
                    "api_calls": calls,
                    "timing": {
                        "manifest_and_input_loading_seconds": (
                            loading_seconds
                        ),
                        "prompt_preparation_seconds": prompt_seconds,
                        "api_call_wall_seconds": request.api_call[
                            "api_call_wall_seconds"
                        ],
                        **request_timing,
                        "response_parsing_seconds": parsing_seconds,
                        "validation_and_metrics_seconds": (
                            evaluation_seconds
                        ),
                        "route_rendering_seconds": rendering_seconds,
                        "total_wall_seconds_before_result_write": (
                            elapsed_seconds(total_timer)
                        ),
                    },
                }
            )
        except Exception as exc:
            record = error_record(exc, phase=current_phase)
            errors.append(record)
            if isinstance(record.get("api_call"), dict):
                calls.append(record["api_call"])
            result.update(
                {
                    "api_calls": calls,
                    "timing": {
                        "manifest_and_input_loading_seconds": (
                            loading_seconds
                        ),
                        "prompt_preparation_seconds": prompt_seconds,
                        "total_wall_seconds_before_result_write": (
                            elapsed_seconds(total_timer)
                        ),
                    },
                    "run_summary": summarize_api_calls(calls),
                    "observability": observability.stop(),
                }
            )
            write_json(result_path, result)
            raise SystemExit(
                f"Zero-shot tamamlanamadı: {exc}"
            ) from exc

        result["run_summary"] = summarize_api_calls(calls)
        result["observability"] = observability.stop()
        write_json(result_path, result)
    finally:
        observability.stop()

    print("Birleşik dinamik zero-shot deneyi tamamlandı.")
    print(f"Provider: {provider.provider_id}")
    print(f"Model: {provider.model_alias}")
    print(f"Problem: {problem.name}")
    print(f"Düğüm sayısı: {problem.dimension}")
    print(f"Geçerli rota: {result['validation']['is_valid']}")
    print(f"Mesafe: {result['distance']}")
    gap = result["gap_to_reference_percent"]
    print(
        "Referans gap: "
        f"{'hesaplanamadı' if gap is None else f'%{gap:.4f}'}"
    )
    request_summary = result["observability"]["request_control"]
    print(
        "API denemesi/retry: "
        f"{request_summary['request_attempt_count']}/"
        f"{request_summary['retry_count']}"
    )
    print(
        "Kontrollü bekleme: "
        f"{request_summary['waits']['controlled_wait_seconds']:.4f} sn"
    )
    print(f"Sonuç dosyası: {result_path}")


if __name__ == "__main__":
    main()
