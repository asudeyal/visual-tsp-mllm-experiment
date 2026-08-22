"""Küçük CVRP kesin çözücüsünün testleri."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.exact_solver import (
    METHOD_NAME,
    solve_exact_cvrp,
)
from src.instances import build_capacity_demo_10
from src.validation import evaluate_solution


EXPECTED_OPTIMAL_DISTANCE = (
    419.72784677751633
)


def test_exact_solver_returns_valid_solution() -> None:
    problem = build_capacity_demo_10()

    solution = solve_exact_cvrp(problem)
    evaluation = evaluate_solution(
        problem,
        solution.routes,
    )

    assert solution.method == METHOD_NAME
    assert solution.proven_optimal is True
    assert solution.vehicle_count == 3
    assert solution.route_loads == (6, 6, 6)
    assert evaluation.valid is True
    assert solution.total_distance == pytest.approx(
        evaluation.total_distance
    )

    record = solution.to_dict()

    assert record["proven_optimal"] is True
    assert record["vehicle_count"] == 3
    assert len(record["routes"]) == 3


def test_exact_solver_matches_known_optimum() -> None:
    problem = build_capacity_demo_10()

    solution = solve_exact_cvrp(problem)

    assert solution.total_distance == pytest.approx(
        EXPECTED_OPTIMAL_DISTANCE,
        rel=1e-12,
    )


def test_exact_solver_is_repeatable() -> None:
    problem = build_capacity_demo_10()

    first = solve_exact_cvrp(problem)
    second = solve_exact_cvrp(problem)

    assert first == second


def test_exact_solver_requires_fixed_vehicle_limit() -> None:
    problem = replace(
        build_capacity_demo_10(),
        vehicle_count=None,
    )

    with pytest.raises(
        ValueError,
        match="sabit araç sayısı",
    ):
        solve_exact_cvrp(problem)


def test_exact_solver_enforces_size_limit() -> None:
    problem = build_capacity_demo_10()

    with pytest.raises(
        ValueError,
        match="fazla büyük",
    ):
        solve_exact_cvrp(
            problem,
            maximum_customer_count=8,
        )