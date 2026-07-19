"""Aynı 10 nokta üzerinde Gemini görsel zero-shot deneyini çalıştırır."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.llm_routes import (
    GEMINI_MODEL,
    parse_single_salesman_route,
    request_gemini_zero_shot_route_detailed,
)
from src.experiment_metrics import (
    elapsed_seconds,
    error_record,
    start_timer,
    summarize_api_calls,
    utc_now_iso,
)
from src.output_paths import build_experiment_paths
from src.tsp_core import (
    TSPSolution,
    percentage_gap,
    plot_solution,
    route_distance,
    validate_tsp_route,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Varsayılan: seçilen run-id içindeki baseline_results.json.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Varsayılan: seçilen run-id içindeki baseline/images/points.png.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--run-id",
        help=(
            "Aynı deney çalıştırmasını adlandırır. Verilirse girdiler aynı "
            "run klasöründeki baseline'dan okunur ve sonuç zero_shot'a yazılır."
        ),
    )
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Mevcut zero-shot sonuç JSON'undan kısa özet üretir; Gemini API "
            "çağrısı yapmaz."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _error_status_code(error: dict[str, Any]) -> int | None:
    status_code = error.get("status_code")
    if status_code is not None:
        return int(status_code)
    prefix = str(error.get("message", "")).split(maxsplit=1)
    return int(prefix[0]) if prefix and prefix[0].isdigit() else None


def build_zero_shot_summary(
    result: dict[str, Any],
    *,
    source_results: Path,
) -> dict[str, Any]:
    """Ayrıntılı Gemini zero-shot sonucundan kısa özet üretir."""

    errors = [
        {
            "phase": error.get("phase"),
            "type": error.get("type"),
            "status_code": _error_status_code(error),
            "api_call_wall_seconds": error.get("api_call", {}).get(
                "api_call_wall_seconds"
            ),
        }
        for error in result.get("errors", [])
    ]
    status = result.get("status", "completed")
    return {
        "experiment": result.get("experiment"),
        "summary_type": "compact",
        "source_results": str(source_results),
        "status": status,
        "run_id": result.get("run_id"),
        "model": result.get("model"),
        "temperature": result.get("temperature"),
        "coordinates_sent_to_model": result.get("model_input", {}).get(
            "coordinates_sent_to_model"
        ),
        "solution": (
            {
                "route": result.get("route"),
                "is_valid": result.get("validation", {}).get("is_valid"),
                "missing_nodes": result.get("validation", {}).get(
                    "missing_nodes"
                ),
                "repeated_nodes": result.get("validation", {}).get(
                    "repeated_nodes"
                ),
                "distance": result.get("metrics", {}).get("distance"),
                "gap_to_or_tools_percent": result.get("metrics", {}).get(
                    "gap_to_or_tools_percent"
                ),
                "gap_to_exact_percent": result.get("metrics", {}).get(
                    "gap_to_exact_percent"
                ),
            }
            if status != "failed"
            else None
        ),
        "api_summary": result.get("run_summary", {}),
        "timing": result.get("timing", {}),
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    run_started_at_utc = utc_now_iso()
    run_timer = start_timer()
    try:
        paths = build_experiment_paths(args.output_dir, args.run_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    baseline_path = args.baseline or paths.baseline / "baseline_results.json"
    if args.baseline is None and args.run_id is None and not baseline_path.exists():
        legacy_baseline = args.output_dir / "baseline_results.json"
        if legacy_baseline.exists():
            baseline_path = legacy_baseline

    image_path = args.image or paths.baseline / "images" / "points.png"
    if args.image is None and not image_path.exists():
        old_method_image = paths.baseline / "points.png"
        legacy_flat_image = args.output_dir / "points.png"
        if old_method_image.exists():
            image_path = old_method_image
        elif args.run_id is None and legacy_flat_image.exists():
            image_path = legacy_flat_image
    method_output_dir = paths.zero_shot
    method_output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = method_output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    result_path = method_output_dir / "gemini_zero_shot_results.json"
    summary_path = method_output_dir / "gemini_zero_shot_summary.json"

    if args.summary_only:
        if not result_path.exists():
            raise SystemExit(
                f"Özetlenecek Gemini zero-shot sonucu bulunamadı: {result_path}"
            )
        existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        write_json(
            summary_path,
            build_zero_shot_summary(existing_result, source_results=result_path),
        )
        print("Gemini zero-shot kısa özeti oluşturuldu; API çağrısı yapılmadı.")
        print(f"Özet dosyası: {summary_path}")
        return

    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit(
            "GEMINI_API_KEY tanımlı değil. Anahtarı koda yazmayın; PowerShell "
            "terminalinde $env:GEMINI_API_KEY ortam değişkeni olarak ayarlayın."
        )
    if not baseline_path.exists() or not image_path.exists():
        raise SystemExit(
            "Baseline dosyaları bulunamadı. Önce "
            "`python run_baseline.py --seed 42 --ortools-time-limit 2` çalıştırın."
        )

    input_timer = start_timer()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    locations = [tuple(point) for point in baseline["locations"]]
    exact_distance = float(baseline["solutions"]["exact"]["distance"])
    or_tools_distance = float(baseline["solutions"]["or_tools"]["distance"])
    input_loading_seconds = elapsed_seconds(input_timer)

    try:
        gemini_result = request_gemini_zero_shot_route_detailed(
            image_path,
            model=args.model,
            temperature=0.0,
        )
    except Exception as exc:
        failure = {
            "experiment": "gemini_visual_zero_shot_tsp",
            "run_id": args.run_id,
            "method": "zero_shot",
            "model": args.model,
            "status": "failed",
            "started_at_utc": run_started_at_utc,
            "finished_at_utc": utc_now_iso(),
            "total_wall_seconds": elapsed_seconds(run_timer),
            "errors": [error_record(exc, phase="zero_shot_initializer")],
        }
        write_json(result_path, failure)
        write_json(
            summary_path,
            build_zero_shot_summary(failure, source_results=result_path),
        )
        raise

    raw_response = gemini_result.text
    parse_timer = start_timer()
    route = parse_single_salesman_route(raw_response)
    response_parsing_seconds = elapsed_seconds(parse_timer)

    evaluation_timer = start_timer()
    validation = validate_tsp_route(route, num_locations=len(locations))

    legal_node_ids = all(0 <= node < len(locations) for node in route)
    distance = route_distance(locations, route) if legal_node_ids else None
    metrics = {
        "distance": distance,
        "gap_to_or_tools_percent": (
            percentage_gap(distance, or_tools_distance) if distance is not None else None
        ),
        "gap_to_exact_percent": (
            percentage_gap(distance, exact_distance) if distance is not None else None
        ),
    }
    validation_and_metrics_seconds = elapsed_seconds(evaluation_timer)

    route_rendering_seconds = 0.0
    if legal_node_ids and len(route) >= 2:
        render_timer = start_timer()
        display_solution = TSPSolution(
            method=f"{args.model}_zero_shot",
            route=route,
            distance=float(distance),
            validation=validation,
        )
        plot_solution(
            locations,
            display_solution,
            image_dir / "gemini_zero_shot_route.png",
        )
        route_rendering_seconds = elapsed_seconds(render_timer)

    total_wall_seconds = elapsed_seconds(run_timer)
    request_preparation_seconds = float(
        gemini_result.api_call.get("request_preparation_seconds", 0.0)
    )
    tracked_seconds = (
        input_loading_seconds
        + request_preparation_seconds
        + float(gemini_result.api_call["api_call_wall_seconds"])
        + response_parsing_seconds
        + validation_and_metrics_seconds
        + route_rendering_seconds
    )

    result = {
        "experiment": "gemini_visual_zero_shot_tsp",
        "run_id": args.run_id,
        "method": "zero_shot",
        "model": args.model,
        "temperature": 0.0,
        "num_locations_including_depot": len(locations),
        "num_salesmen": 1,
        "model_input": {
            "image": str(image_path),
            "coordinates_sent_to_model": False,
        },
        "raw_response": raw_response,
        "route": route,
        "validation": validation.to_dict(),
        "metrics": metrics,
        "api_calls": [gemini_result.api_call],
        "timing": {
            "input_loading_seconds": input_loading_seconds,
            "request_preparation_seconds": request_preparation_seconds,
            "api_call_wall_seconds": gemini_result.api_call[
                "api_call_wall_seconds"
            ],
            "response_parsing_seconds": response_parsing_seconds,
            "validation_and_metrics_seconds": validation_and_metrics_seconds,
            "route_rendering_seconds": route_rendering_seconds,
            "unaccounted_seconds": max(total_wall_seconds - tracked_seconds, 0.0),
            "total_wall_seconds_before_result_write": total_wall_seconds,
        },
        "run_summary": summarize_api_calls([gemini_result.api_call]),
    }

    write_json(result_path, result)
    write_json(
        summary_path,
        build_zero_shot_summary(result, source_results=result_path),
    )

    print("\nGemini zero-shot deneyi tamamlandı.")
    print(f"Model: {args.model}")
    print(f"Ham model cevabı:\n{raw_response}")
    print(f"Ayrıştırılan rota: {route}")
    print(f"Geçerli TSP rotası mı? {validation.is_valid}")
    print(f"Eksik noktalar: {validation.missing_nodes}")
    print(f"Tekrarlanan noktalar: {validation.repeated_nodes}")
    print(f"Gemini rota mesafesi: {distance}")
    print(f"OR-Tools gap: {metrics['gap_to_or_tools_percent']}")
    print(f"Kesin optimum gap: {metrics['gap_to_exact_percent']}")
    print(f"Sonuç dosyası: {result_path}")
    print(f"Kısa özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
