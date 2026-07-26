"""Dinamik TSP problem üretimi/yüklemesi ve OR-Tools baseline deneyi."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.core import (
    evaluate_route,
    method_dir,
    normalize_run_id,
    plot_problem,
    plot_route,
    solve_ortools,
    write_json,
)
from src.metrics import elapsed_seconds, start_timer
from src.problem_cli import (
    add_problem_arguments,
    load_problem_from_args,
)
from src.problem_instance import (
    ReferenceSolution,
    ReferenceType,
)
from src.run_manifest import (
    build_run_manifest,
    snapshot_problem_inputs,
    write_run_manifest,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TSPLIB = ROOT / "data" / "eil51.tsp"
DEFAULT_OPTIMAL_TOUR = ROOT / "data" / "eil51.opt.tour"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_problem_arguments(parser)
    parser.add_argument(
        "--ortools-time-limit",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
    )
    parser.add_argument(
        "--run-id",
        default="dynamic_run_01",
    )
    return parser.parse_args()


def _relative(
    path: Path,
    run_dir: Path,
) -> str:
    return path.resolve().relative_to(
        run_dir.resolve()
    ).as_posix()


def _reference_record(
    problem: Any,
) -> dict[str, Any] | None:
    reference = problem.reference
    if reference is None:
        return None
    evaluation = (
        evaluate_route(problem, reference.route)
        if reference.route is not None
        else None
    )
    return {
        "type": reference.reference_type.value,
        "distance": reference.distance,
        "is_proven_optimal": reference.is_proven_optimal,
        "route": (
            list(reference.route)
            if reference.route is not None
            else None
        ),
        "validation": (
            evaluation["validation"]
            if evaluation is not None
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.ortools_time_limit < 1:
        raise SystemExit("--ortools-time-limit en az 1 olmalıdır.")

    run_id = normalize_run_id(args.run_id)
    run_dir = Path(args.output_dir) / "runs" / run_id
    output = method_dir(
        args.output_dir,
        run_id,
        "baseline",
    )
    result_path = output / "baseline_results.json"
    manifest_path = run_dir / "run_manifest.json"

    total_timer = start_timer()

    load_timer = start_timer()
    problem, input_request = load_problem_from_args(
        args,
        default_tsplib_file=DEFAULT_TSPLIB,
        default_optimal_tour_file=DEFAULT_OPTIMAL_TOUR,
    )
    problem = snapshot_problem_inputs(problem, run_dir)
    loading_seconds = elapsed_seconds(load_timer)

    ortools_timer = start_timer()
    initial_or_tools = solve_ortools(
        problem,
        time_limit_seconds=args.ortools_time_limit,
    )
    ortools_seconds = elapsed_seconds(ortools_timer)

    reference_timer = start_timer()
    if problem.reference is None:
        problem = replace(
            problem,
            reference=ReferenceSolution(
                reference_type=ReferenceType.OR_TOOLS_HEURISTIC,
                distance=float(initial_or_tools["distance"]),
                is_proven_optimal=False,
                route=tuple(initial_or_tools["route"]),
            ),
        )
    or_tools = {
        "method": initial_or_tools["method"],
        **evaluate_route(
            problem,
            initial_or_tools["route"],
        ),
    }
    reference = _reference_record(problem)
    reference_seconds = elapsed_seconds(reference_timer)

    render_timer = start_timer()
    images = output / "images"
    points_image = images / "points.png"
    ortools_image = images / "or_tools_route.png"
    plot_problem(problem, points_image)
    plot_route(
        problem,
        or_tools["route"],
        ortools_image,
        title=(
            f"{problem.name} OR-Tools — "
            f"distance {or_tools['distance']}"
        ),
    )

    reference_image: Path | None = None
    if (
        problem.reference is not None
        and problem.reference.route is not None
        and problem.reference.reference_type
        is not ReferenceType.OR_TOOLS_HEURISTIC
    ):
        reference_image = images / "reference_route.png"
        plot_route(
            problem,
            problem.reference.route,
            reference_image,
            title=(
                f"{problem.name} reference — "
                f"distance {problem.reference.distance}"
            ),
        )
    rendering_seconds = elapsed_seconds(render_timer)

    manifest_timer = start_timer()
    manifest = build_run_manifest(
        run_id=run_id,
        problem=problem,
        run_dir=run_dir,
        input_request={
            key: (
                Path(value).name
                if key in {
                    "instance_file",
                    "optimal_tour_file",
                }
                and value is not None
                else value
            )
            for key, value in input_request.items()
        },
        baseline={
            "method": "or_tools_savings_guided_local_search",
            "time_limit_seconds": args.ortools_time_limit,
            "distance": or_tools["distance"],
            "reference_type": (
                problem.reference.reference_type.value
                if problem.reference is not None
                else None
            ),
        },
    )
    write_run_manifest(manifest_path, manifest)
    manifest_seconds = elapsed_seconds(manifest_timer)

    result = {
        "schema_version": "2.0",
        "experiment": "dynamic_tsp_baseline",
        "run_id": run_id,
        "method": "baseline",
        "problem": {
            "name": problem.name,
            "source_type": problem.source_type.value,
            "dimension": problem.dimension,
            "depot_id": problem.depot_id,
            "edge_weight_type": problem.edge_weight_type,
            "fingerprint_sha256": manifest["problem"][
                "fingerprint_sha256"
            ],
        },
        "reference_solution": reference,
        "or_tools": or_tools,
        "artifacts": {
            "run_manifest": _relative(manifest_path, run_dir),
            "points_image": _relative(points_image, run_dir),
            "or_tools_route_image": _relative(
                ortools_image,
                run_dir,
            ),
            "reference_route_image": (
                _relative(reference_image, run_dir)
                if reference_image is not None
                else None
            ),
        },
        "timing": {
            "problem_loading_seconds": loading_seconds,
            "or_tools_wall_seconds": ortools_seconds,
            "reference_preparation_seconds": reference_seconds,
            "route_rendering_seconds": rendering_seconds,
            "manifest_write_seconds": manifest_seconds,
            "total_wall_seconds_before_result_write": (
                elapsed_seconds(total_timer)
            ),
        },
    }
    write_json(result_path, result)

    print("Dinamik TSP baseline tamamlandı.")
    print(
        f"Problem: {problem.name} "
        f"({problem.dimension} düğüm)"
    )
    print(f"Kaynak: {problem.source_type.value}")
    print(f"Depo: {problem.depot_id}")
    print(f"OR-Tools mesafesi: {or_tools['distance']}")
    if problem.reference is not None:
        print(
            "Referans: "
            f"{problem.reference.reference_type.value}, "
            f"mesafe={problem.reference.distance}, "
            "kanıtlanmış optimum="
            f"{problem.reference.is_proven_optimal}"
        )
    print(f"Manifest: {manifest_path}")
    print(f"Sonuç dosyası: {result_path}")


if __name__ == "__main__":
    main()
