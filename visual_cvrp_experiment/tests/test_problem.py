"""CVRP problem modeli testleri."""

from __future__ import annotations

import pytest

from src.problem import CVRPProblem, Node


def make_problem(
    **overrides: object,
) -> CVRPProblem:
    values: dict[str, object] = {
        "name": "cvrp10_balanced",
        "depot": Node(0, 0.5, 0.5),
        "customers": (
            Node(1, 0.1, 0.1, 1),
            Node(2, 0.2, 0.1, 2),
            Node(3, 0.3, 0.1, 3),
            Node(4, 0.1, 0.8, 1),
            Node(5, 0.2, 0.8, 2),
            Node(6, 0.3, 0.8, 3),
            Node(7, 0.8, 0.2, 1),
            Node(8, 0.8, 0.5, 2),
            Node(9, 0.8, 0.8, 3),
        ),
        "vehicle_capacity": 6,
        "vehicle_count": 3,
    }

    values.update(overrides)

    return CVRPProblem(
        **values,  # type: ignore[arg-type]
    )


def test_balanced_problem_summary() -> None:
    problem = make_problem()

    assert problem.dimension == 10
    assert problem.customer_count == 9
    assert problem.customer_ids == tuple(
        range(1, 10)
    )
    assert problem.total_demand == 18
    assert problem.vehicle_count_lower_bound == 3


def test_distance_uses_euclidean_metric() -> None:
    problem = CVRPProblem(
        name="distance_test",
        depot=Node(0, 0.0, 0.0),
        customers=(
            Node(1, 3.0, 4.0, 1),
        ),
        vehicle_capacity=1,
    )

    assert problem.distance(
        0,
        1,
    ) == pytest.approx(5.0)


def test_to_dict_contains_capacity_metadata() -> None:
    record = make_problem().to_dict()

    assert record["vehicle_capacity"] == 6
    assert record["vehicle_count"] == 3
    assert record["total_demand"] == 18
    assert record["vehicle_count_lower_bound"] == 3
    assert len(record["nodes"]) == 10


def test_duplicate_node_id_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="benzersiz",
    ):
        make_problem(
            customers=(
                Node(1, 0.1, 0.1, 1),
                Node(1, 0.2, 0.2, 2),
            ),
        )


def test_nonzero_depot_demand_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Depo talebi",
    ):
        make_problem(
            depot=Node(0, 0.5, 0.5, 1),
        )


def test_zero_customer_demand_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Müşteri talebi",
    ):
        make_problem(
            customers=(
                Node(1, 0.1, 0.1, 0),
            ),
        )


def test_customer_demand_above_capacity_is_rejected(
) -> None:
    with pytest.raises(
        ValueError,
        match="araç kapasitesini aşamaz",
    ):
        make_problem(
            customers=(
                Node(1, 0.1, 0.1, 7),
            ),
            vehicle_capacity=6,
            vehicle_count=None,
        )


def test_insufficient_fixed_fleet_is_rejected(
) -> None:
    with pytest.raises(
        ValueError,
        match="Araç sayısı",
    ):
        make_problem(
            vehicle_count=2,
        )


def test_unknown_node_is_rejected() -> None:
    with pytest.raises(
        KeyError,
        match="Bilinmeyen",
    ):
        make_problem().node(99)