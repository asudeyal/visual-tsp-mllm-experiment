"""Küçük CVRP örnekleri için kesin dinamik programlama çözücüsü."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import inf, isclose
from typing import Any

from .problem import CVRPProblem
from .validation import evaluate_solution


METHOD_NAME = "exact_subset_dynamic_programming"


@dataclass(frozen=True, slots=True)
class ExactCVRPSolution:
    """Kesin CVRP çözücüsünün sonucu."""

    method: str
    proven_optimal: bool
    routes: tuple[tuple[int, ...], ...]
    route_loads: tuple[int, ...]
    vehicle_count: int
    total_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "proven_optimal": self.proven_optimal,
            "routes": [
                list(route)
                for route in self.routes
            ],
            "route_loads": list(
                self.route_loads
            ),
            "vehicle_count": self.vehicle_count,
            "total_distance": self.total_distance,
        }


def _calculate_subset_loads(
    demands: tuple[int, ...],
) -> list[int]:
    subset_count = 1 << len(demands)
    subset_loads = [0] * subset_count

    for mask in range(1, subset_count):
        least_significant_bit = mask & -mask
        customer_index = (
            least_significant_bit.bit_length() - 1
        )
        previous_mask = (
            mask ^ least_significant_bit
        )
        subset_loads[mask] = (
            subset_loads[previous_mask]
            + demands[customer_index]
        )

    return subset_loads


def _calculate_route_tables(
    problem: CVRPProblem,
    customer_ids: tuple[int, ...],
    subset_loads: list[int],
) -> tuple[
    dict[int, float],
    dict[int, int],
    dict[tuple[int, int], int | None],
]:
    customer_count = len(customer_ids)
    subset_count = 1 << customer_count
    depot_id = problem.depot.node_id

    path_costs: dict[
        tuple[int, int],
        float,
    ] = {}
    predecessors: dict[
        tuple[int, int],
        int | None,
    ] = {}

    for customer_index, customer_id in enumerate(
        customer_ids
    ):
        mask = 1 << customer_index
        path_costs[
            (mask, customer_index)
        ] = problem.distance(
            depot_id,
            customer_id,
        )
        predecessors[
            (mask, customer_index)
        ] = None

    for mask in range(1, subset_count):
        if (
            subset_loads[mask]
            > problem.vehicle_capacity
        ):
            continue

        for last_index in range(customer_count):
            last_bit = 1 << last_index
            if not mask & last_bit:
                continue

            previous_mask = mask ^ last_bit
            if previous_mask == 0:
                continue

            best_cost = inf
            best_predecessor: int | None = None

            for previous_index in range(
                customer_count
            ):
                previous_bit = 1 << previous_index
                if not previous_mask & previous_bit:
                    continue

                candidate_cost = (
                    path_costs[
                        (
                            previous_mask,
                            previous_index,
                        )
                    ]
                    + problem.distance(
                        customer_ids[previous_index],
                        customer_ids[last_index],
                    )
                )

                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_predecessor = (
                        previous_index
                    )

            path_costs[
                (mask, last_index)
            ] = best_cost
            predecessors[
                (mask, last_index)
            ] = best_predecessor

    route_costs: dict[int, float] = {}
    route_end_indices: dict[int, int] = {}

    for mask in range(1, subset_count):
        if (
            subset_loads[mask]
            > problem.vehicle_capacity
        ):
            continue

        candidates = [
            (
                path_costs[
                    (mask, last_index)
                ]
                + problem.distance(
                    customer_ids[last_index],
                    depot_id,
                ),
                last_index,
            )
            for last_index in range(
                customer_count
            )
            if mask & (1 << last_index)
        ]

        best_cost, best_last_index = min(
            candidates
        )
        route_costs[mask] = best_cost
        route_end_indices[mask] = (
            best_last_index
        )

    return (
        route_costs,
        route_end_indices,
        predecessors,
    )


def _partition_customers(
    *,
    full_mask: int,
    vehicle_limit: int,
    vehicle_capacity: int,
    subset_loads: list[int],
    route_costs: dict[int, float],
) -> tuple[int, ...]:
    @lru_cache(maxsize=None)
    def search(
        remaining_mask: int,
        vehicles_left: int,
    ) -> tuple[float, tuple[int, ...]]:
        if remaining_mask == 0:
            return 0.0, ()

        if vehicles_left == 0:
            return inf, ()

        minimum_required_vehicles = (
            subset_loads[remaining_mask]
            + vehicle_capacity
            - 1
        ) // vehicle_capacity

        if (
            minimum_required_vehicles
            > vehicles_left
        ):
            return inf, ()

        first_customer_bit = (
            remaining_mask
            & -remaining_mask
        )

        best_cost = inf
        best_partition: tuple[int, ...] = ()

        subset_mask = remaining_mask
        while subset_mask:
            if (
                subset_mask & first_customer_bit
                and subset_mask in route_costs
            ):
                next_mask = (
                    remaining_mask ^ subset_mask
                )
                (
                    remaining_cost,
                    remaining_partition,
                ) = search(
                    next_mask,
                    vehicles_left - 1,
                )

                if remaining_cost != inf:
                    candidate_cost = (
                        route_costs[subset_mask]
                        + remaining_cost
                    )
                    candidate_partition = (
                        (subset_mask,)
                        + remaining_partition
                    )

                    is_better = (
                        candidate_cost
                        < best_cost - 1e-12
                    )
                    is_equal_and_deterministic = (
                        isclose(
                            candidate_cost,
                            best_cost,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        and (
                            not best_partition
                            or candidate_partition
                            < best_partition
                        )
                    )

                    if (
                        is_better
                        or is_equal_and_deterministic
                    ):
                        best_cost = candidate_cost
                        best_partition = (
                            candidate_partition
                        )

            subset_mask = (
                subset_mask - 1
            ) & remaining_mask

        return best_cost, best_partition

    best_cost, best_partition = search(
        full_mask,
        vehicle_limit,
    )

    if best_cost == inf:
        raise ValueError(
            "Problem verilen araç kapasitesi ve "
            "araç sayısıyla çözülemiyor."
        )

    return best_partition


def _reconstruct_route(
    *,
    problem: CVRPProblem,
    subset_mask: int,
    customer_ids: tuple[int, ...],
    route_end_indices: dict[int, int],
    predecessors: dict[
        tuple[int, int],
        int | None,
    ],
) -> tuple[int, ...]:
    depot_id = problem.depot.node_id
    current_mask = subset_mask
    current_index: int | None = (
        route_end_indices[subset_mask]
    )
    reversed_customer_ids: list[int] = []

    while current_index is not None:
        reversed_customer_ids.append(
            customer_ids[current_index]
        )
        previous_index = predecessors[
            (current_mask, current_index)
        ]
        current_mask ^= 1 << current_index
        current_index = previous_index

    ordered_customer_ids = tuple(
        reversed(reversed_customer_ids)
    )

    return (
        depot_id,
        *ordered_customer_ids,
        depot_id,
    )


def solve_exact_cvrp(
    problem: CVRPProblem,
    *,
    maximum_customer_count: int = 15,
) -> ExactCVRPSolution:
    """Küçük bir CVRP örneğini kesin olarak çöz."""

    if maximum_customer_count < 1:
        raise ValueError(
            "Azami müşteri sayısı pozitif olmalıdır."
        )

    if problem.vehicle_count is None:
        raise ValueError(
            "Kesin çözücü sabit araç sayısı "
            "tanımlanmasını gerektirir."
        )

    if (
        problem.customer_count
        > maximum_customer_count
    ):
        raise ValueError(
            "Problem kesin çözücü için fazla büyük: "
            f"{problem.customer_count} müşteri, "
            f"sınır={maximum_customer_count}."
        )

    customer_ids = problem.customer_ids
    demands = tuple(
        customer.demand
        for customer in problem.customers
    )
    full_mask = (
        1 << problem.customer_count
    ) - 1

    subset_loads = _calculate_subset_loads(
        demands
    )
    (
        route_costs,
        route_end_indices,
        predecessors,
    ) = _calculate_route_tables(
        problem,
        customer_ids,
        subset_loads,
    )

    route_masks = _partition_customers(
        full_mask=full_mask,
        vehicle_limit=problem.vehicle_count,
        vehicle_capacity=(
            problem.vehicle_capacity
        ),
        subset_loads=subset_loads,
        route_costs=route_costs,
    )

    routes = tuple(
        _reconstruct_route(
            problem=problem,
            subset_mask=route_mask,
            customer_ids=customer_ids,
            route_end_indices=(
                route_end_indices
            ),
            predecessors=predecessors,
        )
        for route_mask in route_masks
    )

    evaluation = evaluate_solution(
        problem,
        routes,
    )

    if (
        not evaluation.valid
        or evaluation.total_distance is None
    ):
        raise RuntimeError(
            "Kesin çözücü geçerli bir CVRP "
            "çözümü üretemedi."
        )

    route_loads = tuple(
        route.load
        for route in evaluation.routes
    )

    return ExactCVRPSolution(
        method=METHOD_NAME,
        proven_optimal=True,
        routes=routes,
        route_loads=route_loads,
        vehicle_count=len(routes),
        total_distance=(
            evaluation.total_distance
        ),
    )