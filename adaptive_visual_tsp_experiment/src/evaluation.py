"""Passive observer metrics. None of these values are model inputs."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .problem import route_length
from .schemas import ObserverEvaluation, ProblemInstance, ValidationResult
from .search import canonicalize_route


def validate_route(problem: ProblemInstance, route: Iterable[int]) -> ValidationResult:
    route = tuple(route)
    node_set = set(problem.node_ids)
    unknown = tuple(sorted({node for node in route if node not in node_set}))
    renderable = len(route) >= 2 and not unknown

    starts = bool(route) and route[0] == problem.depot
    ends = bool(route) and route[-1] == problem.depot
    closed = len(route) >= 2 and route[0] == route[-1]

    visits = route[:-1] if closed else route
    known_visits = [node for node in visits if node in node_set]
    counts = Counter(known_visits)
    duplicates = tuple(sorted(node for node, count in counts.items() if count > 1))
    missing = tuple(sorted(node_set - set(known_visits)))

    reasons: list[str] = []
    if unknown:
        reasons.append("unknown_nodes")
    if missing:
        reasons.append("missing_nodes")
    if duplicates:
        reasons.append("duplicate_nodes")
    if not starts:
        reasons.append("wrong_start_depot")
    if not ends:
        reasons.append("wrong_end_depot")
    if not closed:
        reasons.append("not_closed")
    if len(route) != problem.dimension + 1:
        reasons.append("wrong_route_length")

    valid = not reasons
    return ValidationResult(
        valid=valid,
        renderable=renderable,
        missing_nodes=missing,
        duplicate_nodes=duplicates,
        unknown_nodes=unknown,
        starts_at_depot=starts,
        ends_at_depot=ends,
        closed_cycle=closed,
        expected_route_length=problem.dimension + 1,
        observed_route_length=len(route),
        reasons=tuple(reasons),
    )


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


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
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def crossing_count(problem: ProblemInstance, route: Iterable[int]) -> int | None:
    route = tuple(route)
    if len(route) < 4 or any(node not in problem.coordinates for node in route):
        return None
    edges = list(zip(route, route[1:]))
    total = 0
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            c, d = edges[j]
            if len({a, b, c, d}) < 4:
                continue
            if _strict_segments_intersect(
                problem.coordinates[a], problem.coordinates[b],
                problem.coordinates[c], problem.coordinates[d],
            ):
                total += 1
    return total


def evaluate_route(problem: ProblemInstance, route: Iterable[int]) -> ObserverEvaluation:
    route = tuple(route)
    validation = validate_route(problem, route)
    if not validation.renderable:
        return ObserverEvaluation(validation, None, None, None, None)

    crossings = crossing_count(problem, route)
    if not validation.valid:
        # An incomplete/duplicate path length is not a valid TSP objective and is
        # intentionally not reported as a comparable tour distance.
        return ObserverEvaluation(validation, None, None, crossings, None)

    distance = route_length(problem, route)
    gap = None
    if problem.reference_optimum and problem.reference_optimum > 0:
        gap = ((distance - problem.reference_optimum) / problem.reference_optimum) * 100.0

    canonical = canonicalize_route(route, problem.depot)
    return ObserverEvaluation(
        validation=validation,
        distance=distance,
        gap_percent=gap,
        crossings=crossings,
        canonical_route=canonical,
    )
