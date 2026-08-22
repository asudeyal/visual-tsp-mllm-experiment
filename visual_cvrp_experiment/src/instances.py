"""Tekrar üretilebilir CVRP araştırma örnekleri."""

from __future__ import annotations

from .problem import CVRPProblem, Node


CAPACITY_DEMO_10_NAME = "capacity_demo_10"

_CUSTOMER_DATA = (
    # node_id, x, y, demand
    (1, 15.0, 82.0, 2),
    (2, 48.0, 90.0, 3),
    (3, 84.0, 80.0, 1),
    (4, 12.0, 48.0, 3),
    (5, 88.0, 52.0, 2),
    (6, 18.0, 16.0, 1),
    (7, 50.0, 10.0, 2),
    (8, 82.0, 18.0, 3),
    (9, 55.0, 65.0, 1),
)


def build_capacity_demo_10() -> CVRPProblem:
    """İlk 10 düğümlü sabit CVRP örneğini oluştur."""

    return CVRPProblem(
        name=CAPACITY_DEMO_10_NAME,
        depot=Node(
            node_id=0,
            x=50.0,
            y=50.0,
            demand=0,
        ),
        customers=tuple(
            Node(
                node_id=node_id,
                x=x,
                y=y,
                demand=demand,
            )
            for node_id, x, y, demand
            in _CUSTOMER_DATA
        ),
        vehicle_capacity=6,
        vehicle_count=3,
    )