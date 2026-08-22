"""CVRP rotaları için deterministik çözüm doğrulaması."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from .problem import CVRPProblem


@dataclass(frozen=True, slots=True)
class RouteEvaluation:
    """Tek bir araç rotasının doğrulama sonucu."""

    route_index: int
    route: tuple[int, ...]
    customer_ids: tuple[int, ...]
    starts_at_depot: bool
    ends_at_depot: bool
    has_internal_depot: bool
    unknown_node_ids: tuple[int, ...]
    load: int
    capacity_excess: int
    distance: float | None
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_index": self.route_index,
            "route": list(self.route),
            "customer_ids": list(self.customer_ids),
            "starts_at_depot": self.starts_at_depot,
            "ends_at_depot": self.ends_at_depot,
            "has_internal_depot": self.has_internal_depot,
            "unknown_node_ids": list(self.unknown_node_ids),
            "load": self.load,
            "capacity_excess": self.capacity_excess,
            "distance": self.distance,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class SolutionEvaluation:
    """Birden fazla araç rotasından oluşan CVRP çözümünün sonucu."""

    valid: bool
    route_count: int
    routes: tuple[RouteEvaluation, ...]
    missing_customer_ids: tuple[int, ...]
    duplicated_customer_ids: tuple[int, ...]
    unknown_node_ids: tuple[int, ...]
    fleet_limit: int | None
    fleet_limit_exceeded: bool
    total_capacity_excess: int
    total_distance: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "route_count": self.route_count,
            "routes": [
                route.to_dict()
                for route in self.routes
            ],
            "missing_customer_ids": list(
                self.missing_customer_ids
            ),
            "duplicated_customer_ids": list(
                self.duplicated_customer_ids
            ),
            "unknown_node_ids": list(
                self.unknown_node_ids
            ),
            "fleet_limit": self.fleet_limit,
            "fleet_limit_exceeded": (
                self.fleet_limit_exceeded
            ),
            "total_capacity_excess": (
                self.total_capacity_excess
            ),
            "total_distance": self.total_distance,
        }


def _evaluate_route(
    problem: CVRPProblem,
    route: Sequence[int],
    route_index: int,
) -> RouteEvaluation:
    """Tek bir araç rotasını değerlendir."""

    route_tuple = tuple(route)
    depot_id = problem.depot.node_id
    known_node_ids = {
        node.node_id
        for node in problem.nodes
    }
    customer_id_set = set(problem.customer_ids)

    starts_at_depot = (
        len(route_tuple) > 0
        and route_tuple[0] == depot_id
    )
    ends_at_depot = (
        len(route_tuple) > 0
        and route_tuple[-1] == depot_id
    )
    has_internal_depot = (
        depot_id in route_tuple[1:-1]
    )

    unknown_node_ids = tuple(
        sorted(
            {
                node_id
                for node_id in route_tuple
                if node_id not in known_node_ids
            }
        )
    )

    visited_customer_ids = tuple(
        node_id
        for node_id in route_tuple
        if node_id in customer_id_set
    )

    load = sum(
        problem.node(node_id).demand
        for node_id in visited_customer_ids
    )
    capacity_excess = max(
        0,
        load - problem.vehicle_capacity,
    )

    distance: float | None = None
    if (
        len(route_tuple) >= 2
        and not unknown_node_ids
    ):
        distance = sum(
            problem.distance(
                route_tuple[index],
                route_tuple[index + 1],
            )
            for index in range(
                len(route_tuple) - 1
            )
        )

    valid = (
        len(route_tuple) >= 3
        and bool(visited_customer_ids)
        and starts_at_depot
        and ends_at_depot
        and not has_internal_depot
        and not unknown_node_ids
        and capacity_excess == 0
    )

    return RouteEvaluation(
        route_index=route_index,
        route=route_tuple,
        customer_ids=visited_customer_ids,
        starts_at_depot=starts_at_depot,
        ends_at_depot=ends_at_depot,
        has_internal_depot=has_internal_depot,
        unknown_node_ids=unknown_node_ids,
        load=load,
        capacity_excess=capacity_excess,
        distance=distance,
        valid=valid,
    )


def evaluate_solution(
    problem: CVRPProblem,
    routes: Sequence[Sequence[int]],
) -> SolutionEvaluation:
    """CVRP çözümündeki bütün rotaları doğrula."""

    route_evaluations = tuple(
        _evaluate_route(
            problem,
            route,
            route_index=index,
        )
        for index, route in enumerate(
            routes,
            start=1,
        )
    )

    visit_counts = Counter(
        customer_id
        for route in route_evaluations
        for customer_id in route.customer_ids
    )

    missing_customer_ids = tuple(
        customer_id
        for customer_id in problem.customer_ids
        if visit_counts[customer_id] == 0
    )
    duplicated_customer_ids = tuple(
        customer_id
        for customer_id in problem.customer_ids
        if visit_counts[customer_id] > 1
    )
    unknown_node_ids = tuple(
        sorted(
            {
                node_id
                for route in route_evaluations
                for node_id in route.unknown_node_ids
            }
        )
    )

    route_count = len(route_evaluations)
    fleet_limit = problem.vehicle_count
    fleet_limit_exceeded = (
        fleet_limit is not None
        and route_count > fleet_limit
    )

    total_capacity_excess = sum(
        route.capacity_excess
        for route in route_evaluations
    )

    if all(
        route.distance is not None
        for route in route_evaluations
    ):
        total_distance: float | None = sum(
            route.distance or 0.0
            for route in route_evaluations
        )
    else:
        total_distance = None

    valid = (
        route_count > 0
        and all(
            route.valid
            for route in route_evaluations
        )
        and not missing_customer_ids
        and not duplicated_customer_ids
        and not unknown_node_ids
        and not fleet_limit_exceeded
    )

    return SolutionEvaluation(
        valid=valid,
        route_count=route_count,
        routes=route_evaluations,
        missing_customer_ids=missing_customer_ids,
        duplicated_customer_ids=duplicated_customer_ids,
        unknown_node_ids=unknown_node_ids,
        fleet_limit=fleet_limit,
        fleet_limit_exceeded=fleet_limit_exceeded,
        total_capacity_excess=total_capacity_excess,
        total_distance=total_distance,
    )