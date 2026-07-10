"""Aynı 10 nokta üzerinde Gemini görsel zero-shot deneyini çalıştırır."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.llm_routes import (
    GEMINI_MODEL,
    parse_single_salesman_route,
    request_gemini_zero_shot_route,
)
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
        default=Path("output/baseline_results.json"),
    )
    parser.add_argument("--image", type=Path, default=Path("output/points.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model", default=GEMINI_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit(
            "GEMINI_API_KEY tanımlı değil. Anahtarı koda yazmayın; PowerShell "
            "terminalinde $env:GEMINI_API_KEY ortam değişkeni olarak ayarlayın."
        )
    if not args.baseline.exists() or not args.image.exists():
        raise SystemExit(
            "Baseline dosyaları bulunamadı. Önce "
            "`python run_baseline.py --seed 42 --ortools-time-limit 2` çalıştırın."
        )

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    locations = [tuple(point) for point in baseline["locations"]]
    exact_distance = float(baseline["solutions"]["exact"]["distance"])
    or_tools_distance = float(baseline["solutions"]["or_tools"]["distance"])

    raw_response = request_gemini_zero_shot_route(
        args.image,
        model=args.model,
        temperature=0.0,
    )
    route = parse_single_salesman_route(raw_response)
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

    result = {
        "experiment": "gemini_visual_zero_shot_tsp",
        "method": "zero_shot",
        "model": args.model,
        "temperature": 0.0,
        "num_locations_including_depot": len(locations),
        "num_salesmen": 1,
        "model_input": {
            "image": str(args.image),
            "coordinates_sent_to_model": False,
        },
        "raw_response": raw_response,
        "route": route,
        "validation": validation.to_dict(),
        "metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "gemini_zero_shot_results.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if legal_node_ids and len(route) >= 2:
        display_solution = TSPSolution(
            method=f"{args.model}_zero_shot",
            route=route,
            distance=float(distance),
            validation=validation,
        )
        plot_solution(
            locations,
            display_solution,
            args.output_dir / "gemini_zero_shot_route.png",
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


if __name__ == "__main__":
    main()
