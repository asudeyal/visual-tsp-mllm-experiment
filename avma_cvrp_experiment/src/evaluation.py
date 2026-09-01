"""Passive observer metrics for CVRP.

Includes capacity validation, node coverage checks, multi-route evaluation,
and route-crossing metrics.

None of these values are model inputs (Information Firewall).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .problem import multi_route_length
from .schemas import ObserverEvaluation, ProblemInstance, ValidationResult
from .search import canonicalize_route


def validate_cvrp_routes(
    problem: ProblemInstance,
    routes: Sequence[Iterable[int]],
) -> ValidationResult:
    """Validate a multi-vehicle CVRP solution.

    Checks:
    - customer coverage
    - duplicate customers
    - unknown nodes
    - depot at route start/end
    - depot not appearing inside a route
    - closed routes
    - vehicle capacity
    - maximum vehicle count

    All values returned here are observer-side values and are not intended
    to be exposed to the model as numerical optimization feedback.
    """

    all_visits: list[int] = []
    unknown_set: set[int] = set()

    customer_set = set(problem.node_ids) - {problem.depot}

    reasons: list[str] = []

    starts_correct = True
    ends_correct = True
    closed_cycles = True

    parsed_subroutes: list[tuple[int, ...]] = []

    route_loads: list[int] = []
    route_capacity_ratios: list[float] = []
    capacity_exceeded_route_indices: list[int] = []

    # ------------------------------------------------------------------
    # Route-count / fleet-limit validation
    # ------------------------------------------------------------------

    vehicle_count = len(routes)

    if (
        problem.max_vehicles is not None
        and vehicle_count > problem.max_vehicles
    ):
        reasons.append("max_vehicles_exceeded")

    # ------------------------------------------------------------------
    # Route-by-route validation
    # ------------------------------------------------------------------

    for route_index, sub in enumerate(routes):
        sub_t = tuple(sub)
        parsed_subroutes.append(sub_t)

        # Empty / one-node routes cannot form a valid depot-to-depot cycle.
        if len(sub_t) < 2:
            starts_correct = False
            ends_correct = False
            closed_cycles = False

            route_loads.append(0)
            route_capacity_ratios.append(0.0)
            continue

        # Start depot
        if sub_t[0] != problem.depot:
            starts_correct = False

        # End depot
        if sub_t[-1] != problem.depot:
            ends_correct = False

        # Closed cycle
        if sub_t[0] != sub_t[-1]:
            closed_cycles = False

        # --------------------------------------------------------------
        # Customer portion of the route
        # --------------------------------------------------------------
        #
        # For a correctly closed route:
        #
        #     [depot, c1, c2, ..., cn, depot]
        #
        # only c1...cn are customers.
        #
        # If the route is malformed, we still try to inspect its
        # non-depot nodes so that the validator can report useful
        # structural errors.
        # --------------------------------------------------------------

        if (
            len(sub_t) >= 2
            and sub_t[0] == problem.depot
            and sub_t[-1] == problem.depot
        ):
            customers = sub_t[1:-1]
        else:
            customers = tuple(
                node
                for node in sub_t
                if node != problem.depot
            )

        # Depot must never occur inside the customer sequence.
        if problem.depot in customers:
            reasons.append("depot_in_route")

        # --------------------------------------------------------------
        # Unknown nodes
        # --------------------------------------------------------------

        for node in customers:
            if node not in customer_set:
                unknown_set.add(node)
            else:
                all_visits.append(node)

        # --------------------------------------------------------------
        # Capacity
        # --------------------------------------------------------------

        route_load = sum(
            problem.demands.get(node, 0)
            for node in customers
            if node in customer_set
        )

        route_loads.append(route_load)

        if problem.capacity > 0:
            route_capacity_ratios.append(
                route_load / problem.capacity
            )
        else:
            route_capacity_ratios.append(0.0)

        if route_load > problem.capacity:
            capacity_exceeded_route_indices.append(route_index)
            reasons.append("capacity_exceeded")

    # ------------------------------------------------------------------
    # Unknown nodes
    # ------------------------------------------------------------------

    unknown_nodes = tuple(sorted(unknown_set))

    if unknown_nodes:
        reasons.append("unknown_nodes")

    # ------------------------------------------------------------------
    # Customer visit frequencies
    # ------------------------------------------------------------------

    counts = Counter(all_visits)

    duplicates = tuple(
        sorted(
            node
            for node, count in counts.items()
            if count > 1
        )
    )

    missing = tuple(
        sorted(
            customer_set - set(counts.keys())
        )
    )

    if missing:
        reasons.append("missing_nodes")

    if duplicates:
        reasons.append("duplicate_nodes")

    # ------------------------------------------------------------------
    # Structural route checks
    # ------------------------------------------------------------------

    if not starts_correct:
        reasons.append("wrong_start_depot")

    if not ends_correct:
        reasons.append("wrong_end_depot")

    if not closed_cycles:
        reasons.append("not_closed")

    # ------------------------------------------------------------------
    # Final validity
    # ------------------------------------------------------------------

    valid = not reasons

    # A route set can only be rendered if it contains at least one route
    # and does not contain unknown node IDs.
    renderable = (
        len(parsed_subroutes) > 0
        and not unknown_nodes
    )

    observed_len = sum(
        len(route)
        for route in parsed_subroutes
    )

    expected_len = (
        problem.dimension
        + len(parsed_subroutes)
    )

    return ValidationResult(
        valid=valid,
        renderable=renderable,
        missing_nodes=missing,
        duplicate_nodes=duplicates,
        unknown_nodes=unknown_nodes,
        starts_at_depot=starts_correct,
        ends_at_depot=ends_correct,
        closed_cycle=closed_cycles,
        expected_route_length=expected_len,
        observed_route_length=observed_len,
        reasons=tuple(dict.fromkeys(reasons)),
        route_loads=tuple(route_loads),
        route_capacity_ratios=tuple(route_capacity_ratios),
        vehicle_count=vehicle_count,
        capacity_exceeded_route_indices=tuple(
            capacity_exceeded_route_indices
        ),
    )


def _orientation(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def _strict_segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    return (
        (o1 * o2 < 0.0)
        and (o3 * o4 < 0.0)
    )


def crossing_count_multi(
    problem: ProblemInstance,
    routes: Sequence[Iterable[int]],
) -> int | None:
    """Count strict edge crossings across all routes."""

    all_edges: list[tuple[int, int]] = []

    for route in routes:
        route_t = tuple(route)

        if len(route_t) < 2:
            continue

        if any(
            node not in problem.coordinates
            for node in route_t
        ):
            continue

        all_edges.extend(
            zip(route_t, route_t[1:])
        )

    if not all_edges:
        return None

    total = 0

    for i, (a, b) in enumerate(all_edges):
        for j in range(i + 1, len(all_edges)):
            c, d = all_edges[j]

            # Edges sharing an endpoint are not crossings.
            if len({a, b, c, d}) < 4:
                continue

            if _strict_segments_intersect(
                problem.coordinates[a],
                problem.coordinates[b],
                problem.coordinates[c],
                problem.coordinates[d],
            ):
                total += 1

    return total


def evaluate_cvrp_routes(
    problem: ProblemInstance,
    routes: Sequence[Iterable[int]],
) -> ObserverEvaluation:
    """Evaluate a CVRP route set using observer-side metrics."""

    routes_tuple = tuple(
        tuple(route)
        for route in routes
    )

    validation = validate_cvrp_routes(
        problem,
        routes_tuple,
    )

    if not validation.renderable:
        return ObserverEvaluation(
            validation,
            None,
            None,
            None,
            None,
        )

    crossings = crossing_count_multi(
        problem,
        routes_tuple,
    )

    # Do not report distance for an invalid CVRP solution.
    if not validation.valid:
        return ObserverEvaluation(
            validation,
            None,
            None,
            crossings,
            None,
        )

    distance = multi_route_length(
        problem,
        routes_tuple,
    )

    gap = None

    if (
        problem.reference_optimum is not None
        and problem.reference_optimum > 0
    ):
        gap = (
            (distance - problem.reference_optimum)
            / problem.reference_optimum
        ) * 100.0

    canonical_subroutes = tuple(
        canonicalize_route(
            route,
            problem.depot,
        )
        for route in routes_tuple
    )

    return ObserverEvaluation(
        validation=validation,
        distance=distance,
        gap_percent=gap,
        crossings=crossings,
        canonical_routes=canonical_subroutes,
    )