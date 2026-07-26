"""Run manifestindeki görsel TSP problemi için Gemini zero-shot deneyi."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core import (
    evaluate_route,
    method_dir,
    normalize_run_id,
    plot_route,
    write_json,
)
from src.gemini import (
    GEMINI_MODEL,
    initializer_prompt,
    parse_route,
    request_route,
)
from src.metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
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
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Manifest ve promptu API çağrısı yapmadan doğrular.",
    )
    return parser.parse_args()


def _relative(
    path: Path,
    run_dir: Path,
) -> str:
    return path.resolve().relative_to(
        run_dir.resolve()
    ).as_posix()


def main() -> None:
    args = parse_args()
    run_id = normalize_run_id(args.run_id)
    run_dir = Path(args.output_dir) / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    output = method_dir(
        args.output_dir,
        run_id,
        "zero_shot",
    )
    result_path = output / "zero_shot_results.json"

    loading_timer = start_timer()
    if not manifest_path.exists():
        raise SystemExit(
            "Run manifesti bulunamadı. Önce aynı --run-id ile "
            "run_baseline.py çalıştırılmalıdır."
        )
    manifest, problem = load_run_problem(manifest_path)
    points_image = run_dir / "baseline" / "images" / "points.png"
    if not points_image.exists():
        raise SystemExit(
            f"Baseline problem görseli bulunamadı: {points_image}"
        )
    loading_seconds = elapsed_seconds(loading_timer)

    prompt_timer = start_timer()
    prompt = initializer_prompt(problem)
    prompt_seconds = elapsed_seconds(prompt_timer)

    if args.validate_only:
        print("Zero-shot çevrimdışı doğrulaması başarılı.")
        print(f"Run ID: {run_id}")
        print(f"Problem: {problem.name}")
        print(f"Düğüm sayısı: {problem.dimension}")
        print(f"Depo: {problem.depot_id}")
        print(f"Model: {args.model}")
        print(
            "Problem fingerprint: "
            f"{manifest['problem']['fingerprint_sha256']}"
        )
        print(f"Prompt karakter sayısı: {len(prompt)}")
        print("API çağrısı yapılmadı ve kota kullanılmadı.")
        return

    total_timer = start_timer()
    calls: list[dict] = []
    errors: list[dict] = []
    result: dict = {
        "schema_version": "2.0",
        "experiment": "dynamic_visual_zero_shot_tsp",
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
            "provider": "google_gemini",
            "name": args.model,
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
        request = request_route(
            points_image,
            prompt=prompt,
            model=args.model,
            temperature=0.0,
            phase="route_generation",
        )
        calls.append(request.api_call)

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
        image_path = output / "images" / "route.png"
        plot_route(
            problem,
            route,
            image_path,
            title=(
                f"Gemini zero-shot {problem.name} — "
                f"distance {evaluation['distance']}"
            ),
        )
        rendering_seconds = elapsed_seconds(rendering_timer)

        result.update(
            {
                "raw_response": request.text,
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
            }
        )
        result["run_summary"] = summarize_api_calls(calls)
        write_json(result_path, result)
        raise SystemExit(
            f"Zero-shot tamamlanamadı: {exc}"
        ) from exc

    result["run_summary"] = summarize_api_calls(calls)
    write_json(result_path, result)

    print("Gemini dinamik zero-shot deneyi tamamlandı.")
    print(f"Problem: {problem.name}")
    print(f"Düğüm sayısı: {problem.dimension}")
    print(f"Geçerli rota: {result['validation']['is_valid']}")
    print(f"Mesafe: {result['distance']}")
    gap = result["gap_to_reference_percent"]
    print(
        "Referans gap: "
        f"{'hesaplanamadı' if gap is None else f'%{gap:.4f}'}"
    )
    print(f"Sonuç dosyası: {result_path}")


if __name__ == "__main__":
    main()
