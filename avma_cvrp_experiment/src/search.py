"""Route structure utilities used for passive analysis and structural stagnation in CVRP."""

from __future__ import annotations

from itertools import pairwise
from typing import Iterable

from .schemas import StructuralStagnationResult


def canonicalize_route(route: Iterable[int], depot: int) -> tuple[int, ...]:
    route = tuple(route)
    if not route:
        return route
    core = list(route[:-1] if len(route) > 1 and route[0] == route[-1] else route)
    if depot not in core:
        return tuple(route)

    index = core.index(depot)
    forward = core[index:] + core[:index]
    reverse_tail = list(reversed(forward[1:]))
    reverse = [depot, *reverse_tail]
    chosen = min(tuple(forward), tuple(reverse))
    return (*chosen, depot)


def canonicalize_routes(routes: Iterable[Iterable[int]], depot: int) -> tuple[tuple[int, ...], ...]:
    """Canonicalize each sub-route and sort the collection to ensure order independence."""
    canonical = [canonicalize_route(r, depot) for r in routes if len(tuple(r)) > 1]
    return tuple(sorted(canonical))


def undirected_edge_set(routes: Iterable[Iterable[int]]) -> frozenset[tuple[int, int]]:
    """Extract all unique undirected edges across all CVRP sub-routes."""
    edges = set()
    for route in routes:
        r = tuple(route)
        if len(r) >= 2:
            edges.update(tuple(sorted((a, b))) for a, b in zip(r, r[1:]) if a != b)
    return frozenset(edges)


def edge_similarity(routes_a: Iterable[Iterable[int]], routes_b: Iterable[Iterable[int]]) -> float:
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
    recent = history[-window:]
    if len(recent) < window:
        return StructuralStagnationResult(False, len(recent), len(recent), False, 0.0, False)

    canonical = [canonicalize_routes(routes, depot) for routes in recent]
    unique_count = len(set(canonical))
    exact_repeat_signal = unique_count <= max_unique_routes

    similarities = [edge_similarity(a, b) for a, b in pairwise(recent)]
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    similarity_signal = mean_similarity >= similarity_threshold

    return StructuralStagnationResult(
        stagnated=exact_repeat_signal or similarity_signal,
        window_size=len(recent),
        unique_canonical_routes=unique_count,
        exact_repeat_signal=exact_repeat_signal,
        mean_consecutive_similarity=mean_similarity,
        similarity_signal=similarity_signal,
    )


def is_exact_two_opt_transition(
    old_routes: Iterable[Iterable[int]], 
    new_routes: Iterable[Iterable[int]]
) -> bool:
    """Return True when new_routes differs by exactly one structural 2-edge modification.

    For CVRP, a valid structural modification (intra-route 2-opt or inter-route 
    edge exchange) removes exactly 2 edges and adds exactly 2 new edges. 
    This is an audit only. It never repairs or applies the move for the model.
    """
    old_edges = undirected_edge_set(old_routes)
    new_edges = undirected_edge_set(new_routes)
    
    removed = old_edges - new_edges
    added = new_edges - old_edges
    
    return len(removed) == 2 and len(added) == 2