"""TSPLIB eil51 bilinen optimum ve OR-Tools baseline deneyi."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core import (
    KNOWN_OPTIMUM,
    evaluate_route,
    method_dir,
    normalize_run_id,
    parse_tsplib,
    parse_tsplib_tour,
    plot_problem,
    plot_route,
    read_json,
    solve_ortools,
    write_json,
)
from src.metrics import elapsed_seconds, start_timer
from src.summaries import baseline_summary


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=ROOT / "data/eil51.tsp")
    parser.add_argument(
        "--optimal-tour", type=Path, default=ROOT / "data/eil51.opt.tour"
    )
    parser.add_argument("--ortools-time-limit", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--run-id", default="eil51_run_01")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = normalize_run_id(args.run_id)
    output = method_dir(args.output_dir, run_id, "baseline")
    result_path = output / "baseline_results.json"
    summary_path = output / "baseline_summary.json"
    if args.summary_only:
        write_json(summary_path, baseline_summary(read_json(result_path)))
        print(f"Özet dosyası: {summary_path}")
        return

    total_timer = start_timer()
    load_timer = start_timer()
    instance = parse_tsplib(args.instance)
    optimum_route = parse_tsplib_tour(args.optimal_tour)
    known_optimum = {"source": "TSPLIB_known_optimal_tour", **evaluate_route(instance, optimum_route)}
    loading_seconds = elapsed_seconds(load_timer)
    if not known_optimum["validation"]["is_valid"]:
        raise RuntimeError("Paketlenmiş eil51 optimum turu geçerli değil.")
    if known_optimum["distance"] != KNOWN_OPTIMUM:
        raise RuntimeError(
            f"Optimum tur mesafesi {KNOWN_OPTIMUM} yerine {known_optimum['distance']} çıktı."
        )

    ortools_timer = start_timer()
    or_tools = solve_ortools(instance, time_limit_seconds=args.ortools_time_limit)
    ortools_seconds = elapsed_seconds(ortools_timer)

    render_timer = start_timer()
    images = output / "images"
    points_image = images / "points.png"
    exact_image = images / "known_optimum_route.png"
    ortools_image = images / "or_tools_route.png"
    plot_problem(instance, points_image)
    plot_route(
        instance,
        optimum_route,
        exact_image,
        title=f"eil51 known optimum — distance {known_optimum['distance']}",
    )
    plot_route(
        instance,
        or_tools["route"],
        ortools_image,
        title=f"eil51 OR-Tools — distance {or_tools['distance']}",
    )
    render_seconds = elapsed_seconds(render_timer)
    result = {
        "experiment": "tsplib_eil51_baseline",
        "run_id": run_id,
        "instance": {
            "name": instance.name,
            "dimension": instance.dimension,
            "depot_id": 1,
            "edge_weight_type": instance.edge_weight_type,
            "known_optimum_distance": KNOWN_OPTIMUM,
            "instance_file": str(args.instance),
            "optimal_tour_file": str(args.optimal_tour),
            "coordinates": [
                {"node_id": node, "x": xy[0], "y": xy[1]}
                for node, xy in instance.coordinates.items()
            ],
        },
        "known_optimum": known_optimum,
        "or_tools": or_tools,
        "images": {
            "points": str(points_image),
            "known_optimum_route": str(exact_image),
            "or_tools_route": str(ortools_image),
        },
        "timing": {
            "input_loading_seconds": loading_seconds,
            "or_tools_wall_seconds": ortools_seconds,
            "route_rendering_seconds": render_seconds,
            "total_wall_seconds_before_result_write": elapsed_seconds(total_timer),
        },
    }
    write_json(result_path, result)
    write_json(summary_path, baseline_summary(result))
    print("eil51 baseline tamamlandı.")
    print(f"Bilinen optimum: {known_optimum['distance']}")
    print(f"OR-Tools mesafesi: {or_tools['distance']}")
    print(f"OR-Tools gap: %{or_tools['gap_to_known_optimum_percent']:.4f}")
    print(f"Sonuç dosyası: {result_path}")
    print(f"Özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
