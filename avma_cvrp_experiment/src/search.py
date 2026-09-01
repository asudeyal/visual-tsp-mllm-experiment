"""Route structure utilities used for passive analysis and structural stagnation in CVRP."""

from __future__ import annotations

from itertools import pairwise
from typing import Iterable

from .schemas import StructuralStagnationResult


def canonicalize_route(
    route: Iterable[int],
    depot: int,
) -> tuple[int, ...]:
    """Canonicalize a closed route independent of rotation and direction."""
    route = tuple(route)

    if not route:
        return route

    core = list(
        route[:-1]
        if len(route) > 1 and route[0] == route[-1]
        else route
    )

    if depot not in core:
        return tuple(route)

    index = core.index(depot)

    forward = core[index:] + core[:index]

    reverse_tail = list(reversed(forward[1:]))
    reverse = [depot, *reverse_tail]

    chosen = min(
        tuple(forward),
        tuple(reverse),
    )

    return (*chosen, depot)


def canonicalize_routes(
    routes: Iterable[Iterable[int]],
    depot: int,
) -> tuple[tuple[int, ...], ...]:
    """Canonicalize each sub-route and sort the collection to ensure order independence."""
    canonical = [
        canonicalize_route(route, depot)
        for route in routes
        if len(tuple(route)) > 1
    ]

    return tuple(sorted(canonical))


def undirected_edge_set(
    routes: Iterable[Iterable[int]],
) -> frozenset[tuple[int, int]]:
    """Extract all unique undirected edges across all CVRP sub-routes."""
    edges: set[tuple[int, int]] = set()

    for route in routes:
        route_tuple = tuple(route)

        if len(route_tuple) < 2:
            continue

        edges.update(
            tuple(sorted((a, b)))
            for a, b in zip(route_tuple, route_tuple[1:])
            if a != b
        )

    return frozenset(edges)


def edge_similarity(
    routes_a: Iterable[Iterable[int]],
    routes_b: Iterable[Iterable[int]],
) -> float:
    """Return Jaccard similarity of the undirected route-edge sets."""
    a = undirected_edge_set(routes_a)
    b = undirected_edge_set(routes_b)

    if not a and not b:
        return 1.0

    union = a | b

    return len(a & b) / len(union) if union else 1.0


def detect_structural_stagnation(
    history: list[tuple[tuple[int, ...], ...]],
    *,
    depot: int,
    window: int,
    similarity_threshold: float,
    max_unique_routes: int,
) -> StructuralStagnationResult:
    """Detect structural stagnation from recent route history."""
    recent = history[-window:]

    if len(recent) < window:
        return StructuralStagnationResult(
            False,
            len(recent),
            len(recent),
            False,
            0.0,
            False,
        )

    canonical = [
        canonicalize_routes(routes, depot)
        for routes in recent
    ]

    unique_count = len(set(canonical))

    exact_repeat_signal = (
        unique_count <= max_unique_routes
    )

    similarities = [
        edge_similarity(a, b)
        for a, b in pairwise(recent)
    ]

    mean_similarity = (
        sum(similarities) / len(similarities)
        if similarities
        else 0.0
    )

    similarity_signal = (
        mean_similarity >= similarity_threshold
    )

    return StructuralStagnationResult(
        stagnated=(
            exact_repeat_signal
            or similarity_signal
        ),
        window_size=len(recent),
        unique_canonical_routes=unique_count,
        exact_repeat_signal=exact_repeat_signal,
        mean_consecutive_similarity=mean_similarity,
        similarity_signal=similarity_signal,
    )


def _two_opt_variants(
    route: tuple[int, ...],
) -> Iterable[tuple[int, ...]]:
    """Generate all valid intra-route 2-opt variants of a closed route.

    A 2-opt move removes two non-adjacent edges and reconnects the route
    by reversing one contiguous internal segment.

    The depot remains at the beginning and end of the route.
    """
    if len(route) < 5:
        return

    if route[0] != route[-1]:
        return

    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            variant = (
                route[:i]
                + tuple(reversed(route[i : j + 1]))
                + route[j + 1 :]
            )

            if variant != route:
                yield variant


def is_exact_two_opt_transition(
    old_routes: Iterable[Iterable[int]],
    new_routes: Iterable[Iterable[int]],
) -> bool:
    """Return True only for a single intra-route 2-opt transition.

    The transition must satisfy all of the following:

    1. The number of routes is unchanged.
    2. All routes except one remain structurally unchanged.
    3. The changed route is obtained from the old route by reversing
       exactly one contiguous segment.
    4. No inter-route customer exchange is allowed.
    5. The route collection is compared canonically, so vehicle ordering
       and route orientation do not create false differences.

    This function is an audit only. It never repairs or applies a move.
    """
    old = tuple(tuple(route) for route in old_routes)
    new = tuple(tuple(route) for route in new_routes)

    if len(old) != len(new):
        return False

    if not old or not new:
        return False

    old_canonical = canonicalize_routes(
        old,
        depot=old[0][0] if old[0] else 0,
    )

    new_canonical = canonicalize_routes(
        new,
        depot=new[0][0] if new[0] else 0,
    )

    # If canonical structures are already identical, there was no
    # structural 2-opt transition.
    if old_canonical == new_canonical:
        return False

    # Vehicle/order differences must not matter. Therefore we compare
    # each old route against every possible 2-opt variant and then check
    # whether the resulting route collection equals the new collection.
    #
    # We use the depot from the first valid old route. In CVRP all routes
    # are expected to share the same depot.
    depot = None

    for route in old:
        if route:
            depot = route[0]
            break

    if depot is None:
        return False

    target = canonicalize_routes(
        new,
        depot,
    )

    old_canonical_routes = canonicalize_routes(
        old,
        depot,
    )

    for route_index, route in enumerate(old):
        if len(route) < 5:
            continue

        for variant in _two_opt_variants(route):
            candidate_routes = list(old)

            candidate_routes[route_index] = variant

            candidate_canonical = canonicalize_routes(
                candidate_routes,
                depot,
            )

            if candidate_canonical != target:
                continue

            # Ensure that this is genuinely a structural change.
            # This prevents degenerate reversals from being accepted.
            if candidate_canonical == old_canonical_routes:
                continue

            return True

    return False