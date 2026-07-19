"""10 noktalı TSP için ilk aşama: veri, OR-Tools ve kesin optimum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.experiment_metrics import elapsed_seconds, start_timer, utc_now_iso
from src.output_paths import build_experiment_paths
from src.tsp_core import (
    generate_locations,
    percentage_gap,
    plot_problem,
    plot_solution,
    solve_exact_tsp,
    solve_ortools_tsp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ortools-time-limit", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--run-id",
        help=(
            "Aynı deney çalıştırmasını adlandırır. Verilirse çıktılar "
            "output/runs/<run-id>/baseline klasörüne yazılır."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Mevcut baseline sonuç JSON'undan kısa özet üretir; problemi "
            "yeniden çözmez."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_baseline_summary(
    result: dict[str, Any],
    *,
    source_results: Path,
) -> dict[str, Any]:
    """Ayrıntılı baseline sonucundan OR-Tools/kesin çözüm özeti üretir."""

    experiment = result.get("experiment", {})
    solutions = result.get("solutions", {})
    or_tools = solutions.get("or_tools", {})
    exact = solutions.get("exact", {})
    return {
        "experiment": "tsp_baseline",
        "summary_type": "compact",
        "source_results": str(source_results),
        "status": "completed",
        "run_id": experiment.get("run_id"),
        "seed": experiment.get("seed"),
        "num_locations_including_depot": experiment.get(
            "num_locations_including_depot"
        ),
        "or_tools": {
            "route": or_tools.get("route"),
            "distance": or_tools.get("distance"),
            "is_valid": or_tools.get("validation", {}).get("is_valid"),
            "gap_to_exact_percent": result.get("metrics", {}).get(
                "or_tools_gap_to_exact_percent"
            ),
        },
        "exact_brute_force": {
            "route": exact.get("route"),
            "distance": exact.get("distance"),
            "is_valid": exact.get("validation", {}).get("is_valid"),
        },
        "timing": result.get("timing", {}),
    }


def main() -> None:
    args = parse_args()
    run_started_at_utc = utc_now_iso()
    run_timer = start_timer()
    try:
        paths = build_experiment_paths(args.output_dir, args.run_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    method_output_dir = paths.baseline
    method_output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = method_output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    result_path = method_output_dir / "baseline_results.json"
    summary_path = method_output_dir / "baseline_summary.json"

    if args.summary_only:
        if not result_path.exists():
            raise SystemExit(f"Özetlenecek baseline sonucu bulunamadı: {result_path}")
        existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        write_json(
            summary_path,
            build_baseline_summary(existing_result, source_results=result_path),
        )
        print("Baseline kısa özeti oluşturuldu; problem yeniden çözülmedi.")
        print(f"Özet dosyası: {summary_path}")
        return

    phase_timer = start_timer()
    locations = generate_locations(num_locations=10, seed=args.seed)
    problem_generation_seconds = elapsed_seconds(phase_timer)

    phase_timer = start_timer()
    or_tools = solve_ortools_tsp(
        locations, time_limit_seconds=args.ortools_time_limit
    )
    or_tools_seconds = elapsed_seconds(phase_timer)

    phase_timer = start_timer()
    exact = solve_exact_tsp(locations)
    exact_brute_force_seconds = elapsed_seconds(phase_timer)

    phase_timer = start_timer()
    plot_problem(locations, image_dir / "points.png")
    problem_plot_seconds = elapsed_seconds(phase_timer)

    phase_timer = start_timer()
    plot_solution(locations, or_tools, image_dir / "or_tools_route.png")
    or_tools_plot_seconds = elapsed_seconds(phase_timer)

    phase_timer = start_timer()
    plot_solution(locations, exact, image_dir / "exact_route.png")
    exact_plot_seconds = elapsed_seconds(phase_timer)

    result = {
        "experiment": {
            "run_id": args.run_id,
            "num_locations_including_depot": 10,
            "num_salesmen": 1,
            "seed": args.seed,
            "coordinate_distribution": "uniform[0, 5)",
            "ortools_time_limit_seconds": args.ortools_time_limit,
        },
        "locations": locations,
        "solutions": {
            "or_tools": or_tools.to_dict(),
            "exact": exact.to_dict(),
        },
        "metrics": {
            "or_tools_gap_to_exact_percent": percentage_gap(
                or_tools.distance, exact.distance
            )
        },
        "timing": {
            "started_at_utc": run_started_at_utc,
            "finished_at_utc_before_result_write": utc_now_iso(),
            "problem_generation_seconds": problem_generation_seconds,
            "or_tools_seconds": or_tools_seconds,
            "exact_brute_force_seconds": exact_brute_force_seconds,
            "problem_plot_seconds": problem_plot_seconds,
            "or_tools_plot_seconds": or_tools_plot_seconds,
            "exact_plot_seconds": exact_plot_seconds,
            "total_wall_seconds_before_result_write": elapsed_seconds(run_timer),
        },
    }

    write_json(result_path, result)
    write_json(
        summary_path,
        build_baseline_summary(result, source_results=result_path),
    )

    print("10 noktalı TSP başlangıç deneyi tamamlandı.")
    print(f"OR-Tools rota: {or_tools.route}")
    print(f"OR-Tools mesafe: {or_tools.distance:.6f}")
    print(f"Kesin optimum rota: {exact.route}")
    print(f"Kesin optimum mesafe: {exact.distance:.6f}")
    print(
        "OR-Tools optimum gap: "
        f"{result['metrics']['or_tools_gap_to_exact_percent']:.6f}%"
    )
    print(f"Sonuç dosyası: {result_path}")
    print(f"Kısa özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
