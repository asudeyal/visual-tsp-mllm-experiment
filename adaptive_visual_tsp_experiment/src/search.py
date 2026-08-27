"""Route structure utilities used for passive analysis and structural stagnation."""

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


def undirected_edge_set(route: Iterable[int]) -> frozenset[tuple[int, int]]:
    route = tuple(route)
    if len(route) < 2:
        return frozenset()
    return frozenset(tuple(sorted((a, b))) for a, b in zip(route, route[1:]) if a != b)


def edge_similarity(route_a: Iterable[int], route_b: Iterable[int]) -> float:
    a = undirected_edge_set(route_a)
    b = undirected_edge_set(route_b)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def detect_structural_stagnation(
    history: list[tuple[int, ...]],
    *,
    depot: int,
    window: int,
    similarity_threshold: float,
    max_unique_routes: int,
) -> StructuralStagnationResult:
    recent = history[-window:]
    if len(recent) < window:
        return StructuralStagnationResult(False, len(recent), len(recent), False, 0.0, False)

    canonical = [canonicalize_route(route, depot) for route in recent]
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


def is_exact_two_opt_transition(old_route: Iterable[int], new_route: Iterable[int]) -> bool:
    """Return True when new_route differs by exactly one standard 2-opt move.

    This is an audit only. It never repairs or applies the move for the model.
    """
    old = tuple(old_route)
    new = tuple(new_route)
    if len(old) != len(new) or len(old) < 5 or old[0] != old[-1] or new[0] != new[-1]:
        return False
    if old[0] != new[0] or set(old[:-1]) != set(new[:-1]):
        return False

    n = len(old) - 1
    for i in range(0, n - 2):
        for k in range(i + 2, n):
            if i == 0 and k == n - 1:
                continue
            candidate = old[: i + 1] + tuple(reversed(old[i + 1 : k + 1])) + old[k + 1 :]
            if candidate == new:
                return True
    return False
