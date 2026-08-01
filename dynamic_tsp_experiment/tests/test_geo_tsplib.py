from pathlib import Path

import pytest

from src.core import (
    distance_matrix,
    evaluate_route,
    plot_problem,
    plot_route,
    solve_ortools,
)
from src.problem_instance import (
    ProblemSource,
    ReferenceType,
)
from src.problem_loader import (
    geo_coordinate_to_radians,
    geo_distance,
    load_tsplib_problem,
)


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_FILE = ROOT / "data" / "ulysses16.tsp"
OPTIMAL_TOUR_FILE = ROOT / "data" / "ulysses16.opt.tour"


def load_ulysses16():
    return load_tsplib_problem(
        INSTANCE_FILE,
        optimal_tour_file=OPTIMAL_TOUR_FILE,
    )


def test_geo_coordinate_uses_tsplib_degree_minute_conversion() -> None:
    expected_degrees = 38.0 + (5.0 * 0.24 / 3.0)
    expected_radians = expected_degrees * 3.141592653589793 / 180.0

    assert geo_coordinate_to_radians(38.24) == pytest.approx(
        expected_radians
    )


def test_geo_distance_matches_known_ulysses16_edge() -> None:
    first = (38.24, 20.42)
    second = (39.57, 26.15)

    assert geo_distance(first, second) == 509
    assert geo_distance(second, first) == 509


def test_ulysses16_loads_with_proven_optimum() -> None:
    problem = load_ulysses16()

    assert problem.name == "ulysses16"
    assert problem.dimension == 16
    assert problem.node_ids == tuple(range(1, 17))
    assert problem.depot_id == 1
    assert problem.edge_weight_type == "GEO"
    assert problem.source_type is ProblemSource.TSPLIB

    assert problem.reference is not None
    assert (
        problem.reference.reference_type
        is ReferenceType.TSPLIB_KNOWN_OPTIMUM
    )
    assert problem.reference.is_proven_optimal is True
    assert problem.reference.distance == 6859.0
    assert problem.reference.route is not None
    assert problem.reference.route[0] == 1
    assert problem.reference.route[-1] == 1
    assert len(problem.reference.route) == 17


def test_ulysses16_optimal_route_evaluates_to_zero_gap() -> None:
    problem = load_ulysses16()
    assert problem.reference is not None
    assert problem.reference.route is not None

    result = evaluate_route(
        problem,
        problem.reference.route,
    )

    assert result["validation"]["is_valid"] is True
    assert result["distance"] == 6859
    assert result["reference_distance"] == 6859.0
    assert result["gap_to_reference_percent"] == 0.0


def test_geo_distance_matrix_is_integer_and_symmetric() -> None:
    problem = load_ulysses16()
    matrix = distance_matrix(problem)

    assert len(matrix) == 16
    assert all(len(row) == 16 for row in matrix)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            assert value.is_integer()
            assert value == matrix[column_index][row_index]


def test_ortools_returns_valid_geo_route() -> None:
    problem = load_ulysses16()

    result = solve_ortools(
        problem,
        time_limit_seconds=1,
    )

    assert result["validation"]["is_valid"] is True
    assert isinstance(result["distance"], int)
    assert result["distance"] >= 6859


def test_geo_problem_and_route_images_are_created(
    tmp_path: Path,
) -> None:
    problem = load_ulysses16()
    assert problem.reference is not None
    assert problem.reference.route is not None

    points_path = tmp_path / "points.png"
    route_path = tmp_path / "route.png"

    plot_problem(problem, points_path)
    plot_route(
        problem,
        problem.reference.route,
        route_path,
        title="ulysses16 optimum",
    )

    assert points_path.is_file()
    assert points_path.stat().st_size > 0
    assert route_path.is_file()
    assert route_path.stat().st_size > 0
