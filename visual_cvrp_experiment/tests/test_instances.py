"""Sabit CVRP araştırma örneklerinin testleri."""

from __future__ import annotations

from src.instances import (
    CAPACITY_DEMO_10_NAME,
    build_capacity_demo_10,
)
from src.validation import evaluate_solution


def test_capacity_demo_summary() -> None:
    problem = build_capacity_demo_10()

    assert problem.name == CAPACITY_DEMO_10_NAME
    assert problem.dimension == 10
    assert problem.customer_count == 9
    assert problem.depot.node_id == 0
    assert problem.depot.demand == 0
    assert problem.vehicle_capacity == 6
    assert problem.vehicle_count == 3
    assert problem.total_demand == 18
    assert problem.vehicle_count_lower_bound == 3


def test_capacity_demo_has_balanced_demands() -> None:
    problem = build_capacity_demo_10()

    demands = sorted(
        customer.demand
        for customer in problem.customers
    )

    assert demands == [
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]


def test_capacity_demo_is_repeatable() -> None:
    first = build_capacity_demo_10()
    second = build_capacity_demo_10()

    assert first.to_dict() == second.to_dict()


def test_capacity_demo_has_a_feasible_solution() -> None:
    problem = build_capacity_demo_10()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (0, 1, 2, 3, 0),
            (0, 4, 5, 6, 0),
            (0, 7, 8, 9, 0),
        ),
    )

    assert evaluation.valid is True
    assert [
        route.load
        for route in evaluation.routes
    ] == [6, 6, 6]


def test_customer_coordinates_are_unique() -> None:
    problem = build_capacity_demo_10()

    coordinates = {
        (node.x, node.y)
        for node in problem.nodes
    }

    assert len(coordinates) == problem.dimension