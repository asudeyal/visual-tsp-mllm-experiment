"""CVRP çözüm doğrulayıcısının testleri."""

from __future__ import annotations

from src.problem import CVRPProblem, Node
from src.validation import evaluate_solution


def make_problem() -> CVRPProblem:
    return CVRPProblem(
        name="capacity_demo_10",
        depot=Node(
            node_id=0,
            x=0.0,
            y=0.0,
            demand=0,
        ),
        customers=tuple(
            Node(
                node_id=index,
                x=float(index),
                y=float(index % 3),
                demand=((index - 1) % 3) + 1,
            )
            for index in range(1, 10)
        ),
        vehicle_capacity=6,
        vehicle_count=3,
    )


def test_valid_solution_is_accepted() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (0, 1, 2, 3, 0),
            (0, 4, 5, 6, 0),
            (0, 7, 8, 9, 0),
        ),
    )

    assert evaluation.valid is True
    assert evaluation.route_count == 3
    assert evaluation.missing_customer_ids == ()
    assert evaluation.duplicated_customer_ids == ()
    assert evaluation.total_capacity_excess == 0
    assert evaluation.total_distance is not None
    assert [
        route.load
        for route in evaluation.routes
    ] == [6, 6, 6]


def test_capacity_violation_is_reported() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (0, 3, 6, 9, 0),
        ),
    )

    route = evaluation.routes[0]

    assert evaluation.valid is False
    assert route.valid is False
    assert route.load == 9
    assert route.capacity_excess == 3
    assert evaluation.total_capacity_excess == 3


def test_missing_and_duplicate_customers_are_reported() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (0, 1, 2, 3, 0),
            (0, 1, 4, 5, 0),
            (0, 6, 7, 8, 0),
        ),
    )

    assert evaluation.valid is False
    assert evaluation.missing_customer_ids == (9,)
    assert evaluation.duplicated_customer_ids == (1,)


def test_route_without_depot_is_rejected() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (1, 2, 3),
        ),
    )

    route = evaluation.routes[0]

    assert route.starts_at_depot is False
    assert route.ends_at_depot is False
    assert route.valid is False
    assert evaluation.valid is False


def test_internal_depot_is_rejected() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (0, 1, 0, 2, 0),
        ),
    )

    route = evaluation.routes[0]

    assert route.has_internal_depot is True
    assert route.valid is False
    assert evaluation.valid is False


def test_unknown_node_is_reported() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (0, 1, 99, 0),
        ),
    )

    route = evaluation.routes[0]

    assert route.unknown_node_ids == (99,)
    assert route.distance is None
    assert evaluation.unknown_node_ids == (99,)
    assert evaluation.total_distance is None
    assert evaluation.valid is False


def test_fixed_fleet_limit_is_enforced() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (0, 1, 2, 0),
            (0, 3, 4, 0),
            (0, 5, 6, 0),
            (0, 7, 8, 9, 0),
        ),
    )

    assert evaluation.route_count == 4
    assert evaluation.fleet_limit == 3
    assert evaluation.fleet_limit_exceeded is True
    assert evaluation.valid is False


def test_empty_solution_is_rejected() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(),
    )

    assert evaluation.valid is False
    assert evaluation.route_count == 0
    assert evaluation.missing_customer_ids == tuple(
        range(1, 10)
    )
    assert evaluation.total_distance == 0.0


def test_evaluation_can_be_serialized() -> None:
    problem = make_problem()

    evaluation = evaluate_solution(
        problem,
        routes=(
            (0, 1, 2, 3, 0),
            (0, 4, 5, 6, 0),
            (0, 7, 8, 9, 0),
        ),
    )

    record = evaluation.to_dict()

    assert record["valid"] is True
    assert record["route_count"] == 3
    assert record["fleet_limit"] == 3
    assert record["routes"][0]["load"] == 6
    assert record["routes"][0]["route"] == [
        0,
        1,
        2,
        3,
        0,
    ]