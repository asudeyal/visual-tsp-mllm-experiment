from pathlib import Path

from src.core import (
    evaluate_route,
    percentage_gap,
    route_distance,
    solve_ortools,
)
from src.problem_loader import (
    generate_random_problem,
    load_tsplib_problem,
)


ROOT = Path(__file__).resolve().parents[1]


def test_eil51_instance_and_known_optimum() -> None:
    instance = load_tsplib_problem(
        ROOT / "data" / "eil51.tsp",
        optimal_tour_file=ROOT / "data" / "eil51.opt.tour",
    )

    assert instance.reference is not None
    assert instance.reference.route is not None

    result = evaluate_route(
        instance,
        instance.reference.route,
    )

    assert instance.dimension == 51
    assert instance.node_ids == tuple(range(1, 52))
    assert result["validation"]["is_valid"] is True
    assert result["distance"] == 426
    assert result["reference_distance"] == 426.0
    assert result["gap_to_reference_percent"] == 0.0


def test_invalid_route_reports_missing_nodes_and_null_gap() -> None:
    instance = load_tsplib_problem(
        ROOT / "data" / "eil51.tsp",
        optimal_tour_file=ROOT / "data" / "eil51.opt.tour",
    )

    result = evaluate_route(
        instance,
        [1, 2, 3, 1],
    )

    assert result["validation"]["is_valid"] is False
    assert 4 in result["validation"]["missing_nodes"]
    assert result["gap_to_reference_percent"] is None


def test_percentage_gap_requires_explicit_reference() -> None:
    assert percentage_gap(426, 426) == 0.0
    assert abs(percentage_gap(468.6, 426) - 10.0) < 1e-12


def test_random_float_distance_is_not_tsplib_rounded() -> None:
    problem = generate_random_problem(10, seed=42)
    route = [*problem.node_ids, problem.depot_id]

    distance = route_distance(problem, route)

    assert isinstance(distance, float)
    assert distance > 0


def test_ortools_returns_valid_random_route() -> None:
    problem = generate_random_problem(10, seed=42)

    result = solve_ortools(
        problem,
        time_limit_seconds=1,
    )

    assert result["validation"]["is_valid"] is True
    assert result["distance"] is not None
    assert result["gap_to_reference_percent"] is None
