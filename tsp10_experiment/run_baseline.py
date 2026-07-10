"""10 noktalı TSP için ilk aşama: veri, OR-Tools ve kesin optimum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    locations = generate_locations(num_locations=10, seed=args.seed)
    or_tools = solve_ortools_tsp(
        locations, time_limit_seconds=args.ortools_time_limit
    )
    exact = solve_exact_tsp(locations)

    result = {
        "experiment": {
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
    }

    result_path = args.output_dir / "baseline_results.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_problem(locations, args.output_dir / "points.png")
    plot_solution(locations, or_tools, args.output_dir / "or_tools_route.png")
    plot_solution(locations, exact, args.output_dir / "exact_route.png")

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


if __name__ == "__main__":
    main()
