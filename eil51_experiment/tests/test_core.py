from pathlib import Path

from src.core import (
    KNOWN_OPTIMUM,
    evaluate_route,
    parse_tsplib,
    parse_tsplib_tour,
    percentage_gap,
)


ROOT = Path(__file__).resolve().parents[1]


def test_eil51_instance_and_known_optimum() -> None:
    instance = parse_tsplib(ROOT / "data/eil51.tsp")
    route = parse_tsplib_tour(ROOT / "data/eil51.opt.tour")
    result = evaluate_route(instance, route)
    assert instance.dimension == 51
    assert instance.node_ids == list(range(1, 52))
    assert result["validation"]["is_valid"] is True
    assert result["distance"] == KNOWN_OPTIMUM == 426
    assert result["gap_to_known_optimum_percent"] == 0.0


def test_invalid_route_reports_missing_nodes() -> None:
    instance = parse_tsplib(ROOT / "data/eil51.tsp")
    result = evaluate_route(instance, [1, 2, 3, 1])
    assert result["validation"]["is_valid"] is False
    assert 4 in result["validation"]["missing_nodes"]


def test_percentage_gap() -> None:
    assert percentage_gap(426) == 0.0
    assert abs(percentage_gap(468.6) - 10.0) < 1e-12
