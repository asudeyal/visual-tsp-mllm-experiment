"""eil51 görselinden Gemini ile tek çağrıda zero-shot TSP rotası üretir."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core import (
    evaluate_route,
    method_dir,
    normalize_run_id,
    parse_tsplib,
    plot_route,
    read_json,
    write_json,
)
from src.gemini import GEMINI_MODEL, initializer_prompt, parse_route, request_route
from src.metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
)
from src.summaries import zero_shot_summary


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=ROOT / "data/eil51.tsp")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--run-id", default="eil51_run_01")
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = normalize_run_id(args.run_id)
    output = method_dir(args.output_dir, run_id, "zero_shot")
    baseline = method_dir(args.output_dir, run_id, "baseline")
    result_path = output / "gemini_zero_shot_results.json"
    summary_path = output / "gemini_zero_shot_summary.json"
    if args.summary_only:
        write_json(summary_path, zero_shot_summary(read_json(result_path)))
        print(f"Özet dosyası: {summary_path}")
        return

    points_image = baseline / "images/points.png"
    if not points_image.exists():
        raise SystemExit("Önce aynı --run-id ile run_baseline.py çalıştırılmalıdır.")
    instance = parse_tsplib(args.instance)
    total_timer = start_timer()
    calls: list[dict] = []
    errors: list[dict] = []
    result: dict = {
        "experiment": "gemini_visual_zero_shot_eil51",
        "run_id": run_id,
        "method": "zero_shot",
        "model": args.model,
        "temperature": 0.0,
        "num_locations_including_depot": instance.dimension,
        "num_salesmen": 1,
        "model_input": {
            "image": str(points_image),
            "coordinates_sent_to_model": False,
        },
        "errors": errors,
    }
    try:
        request = request_route(
            points_image,
            prompt=initializer_prompt(),
            model=args.model,
            temperature=0.0,
            phase="route_generation",
        )
        calls.append(request.api_call)
        parsing_timer = start_timer()
        route = parse_route(request.text)
        parsing_seconds = elapsed_seconds(parsing_timer)
        evaluation_timer = start_timer()
        evaluation = evaluate_route(instance, route)
        evaluation_seconds = elapsed_seconds(evaluation_timer)
        rendering_timer = start_timer()
        image_path = output / "images/route.png"
        plot_route(
            instance,
            route,
            image_path,
            title=f"Gemini zero-shot eil51 — distance {evaluation['distance']}",
        )
        rendering_seconds = elapsed_seconds(rendering_timer)
        result.update(
            {
                "raw_response": request.text,
                **evaluation,
                "route_image": str(image_path),
                "api_calls": calls,
                "timing": {
                    "api_call_wall_seconds": request.api_call[
                        "api_call_wall_seconds"
                    ],
                    "response_parsing_seconds": parsing_seconds,
                    "validation_and_metrics_seconds": evaluation_seconds,
                    "route_rendering_seconds": rendering_seconds,
                    "total_wall_seconds_before_result_write": elapsed_seconds(
                        total_timer
                    ),
                },
            }
        )
    except Exception as exc:
        record = error_record(exc, phase="route_generation")
        errors.append(record)
        if isinstance(record.get("api_call"), dict):
            calls.append(record["api_call"])
        result.update(
            {
                "api_calls": calls,
                "timing": {
                    "total_wall_seconds_before_result_write": elapsed_seconds(
                        total_timer
                    )
                },
            }
        )
        result["run_summary"] = summarize_api_calls(calls)
        write_json(result_path, result)
        write_json(summary_path, zero_shot_summary(result))
        raise SystemExit(f"Zero-shot tamamlanamadı: {exc}") from exc

    result["run_summary"] = summarize_api_calls(calls)
    write_json(result_path, result)
    write_json(summary_path, zero_shot_summary(result))
    print("Gemini eil51 zero-shot deneyi tamamlandı.")
    print(f"Geçerli rota: {result['validation']['is_valid']}")
    print(f"Mesafe: {result['distance']}")
    gap = result["gap_to_known_optimum_percent"]
    print(f"Bilinen optimum gap: {'hesaplanamadı' if gap is None else f'%{gap:.4f}'}")
    print(f"Sonuç dosyası: {result_path}")
    print(f"Özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
