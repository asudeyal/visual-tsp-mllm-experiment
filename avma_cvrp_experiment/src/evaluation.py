"""Passive observer metrics for CVRP. 
Includes capacity validation, node coverage checks, and multi-route evaluation.
None of these values are model inputs (Information Firewall).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .problem import multi_route_length, route_length
from .schemas import ObserverEvaluation, ProblemInstance, ValidationResult
from .search import canonicalize_route


def validate_cvrp_routes(problem: ProblemInstance, routes: Sequence[Iterable[int]]) -> ValidationResult:
    """CVRP için çoklu rotaların kapasite, depo ve düğüm ziyaret kısıtlarını denetler."""
    all_visits: list[int] = []
    unknown_set: set[int] = set()
    node_set = set(problem.node_ids) - {problem.depot}  # Müşteri düğümleri
    
    reasons: list[str] = []
    starts_correct = True
    ends_correct = True
    closed_cycles = True

    parsed_subroutes = []
    for sub in routes:
        sub_t = tuple(sub)
        parsed_subroutes.append(sub_t)
        
        if len(sub_t) < 2:
            closed_cycles = False
            continue
            
        if sub_t[0] != problem.depot:
            starts_correct = False
        if sub_t[-1] != problem.depot:
            ends_correct = False
        if sub_t[0] != sub_t[-1]:
            closed_cycles = False

        # Alt rota içindeki müşteriler (başlangıç ve bitiş deposu hariç)
        customers = sub_t[:-1] if (len(sub_t) > 1 and sub_t[0] == sub_t[-1] == problem.depot) else sub_t
        
        # Kapasite kontrolü (Her alt rotadaki toplam talep, problem.capacity değerini aşamaz)
        sub_demand = sum(problem.demands.get(node, 0) for node in customers)
        if sub_demand > problem.capacity:
            reasons.append("capacity_exceeded")

        for node in customers:
            if node not in node_set and node != problem.depot:
                unknown_set.add(node)
            else:
                all_visits.append(node)

    unknown_nodes = tuple(sorted(unknown_set))
    if unknown_nodes:
        reasons.append("unknown_nodes")

    # Müşteri ziyaret frekansları (Her müşteri tam bir kez ziyaret edilmeli)
    counts = Counter(all_visits)
    duplicates = tuple(sorted(node for node, count in counts.items() if count > 1))
    missing = tuple(sorted(node_set - set(counts.keys())))

    if missing:
        reasons.append("missing_nodes")
    if duplicates:
        reasons.append("duplicate_nodes")
    if not starts_correct:
        reasons.append("wrong_start_depot")
    if not ends_correct:
        reasons.append("wrong_end_depot")
    if not closed_cycles:
        reasons.append("not_closed")

    valid = not reasons
    renderable = len(parsed_subroutes) > 0 and not unknown_nodes

    observed_len = sum(len(r) for r in parsed_subroutes)
    expected_len = problem.dimension + len(parsed_subroutes)

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


def crossing_count_multi(problem: ProblemInstance, routes: Sequence[Iterable[int]]) -> int | None:
    """Çoklu rotalar üzerindeki toplam kenar kesişim sayısını hesaplar."""
    all_edges = []
    for r in routes:
        route_t = tuple(r)
        if len(route_t) < 4 or any(node not in problem.coordinates for node in route_t):
            continue
        all_edges.extend(list(zip(route_t, route_t[1:])))

    if not all_edges:
        return None

    total = 0
    for i, (a, b) in enumerate(all_edges):
        for j in range(i + 1, len(all_edges)):
            c, d = all_edges[j]
            if len({a, b, c, d}) < 4:
                continue
            if _strict_segments_intersect(
                problem.coordinates[a], problem.coordinates[b],
                problem.coordinates[c], problem.coordinates[d],
            ):
                total += 1
    return total


def evaluate_cvrp_routes(problem: ProblemInstance, routes: Sequence[Iterable[int]]) -> ObserverEvaluation:
    routes_tuple = tuple(tuple(r) for r in routes)
    validation = validate_cvrp_routes(problem, routes_tuple)
    
    if not validation.renderable:
        return ObserverEvaluation(validation, None, None, None, None)

    crossings = crossing_count_multi(problem, routes_tuple)
    if not validation.valid:
        # Geçersiz bir CVRP çözümünün maliyeti observer için raporlanmaz
        return ObserverEvaluation(validation, None, None, crossings, None)

    distance = multi_route_length(problem, routes_tuple)
    gap = None
    if problem.reference_optimum and problem.reference_optimum > 0:
        gap = ((distance - problem.reference_optimum) / problem.reference_optimum) * 100.0

    canonical_subroutes = tuple(canonicalize_route(r, problem.depot) for r in routes_tuple)

    return ObserverEvaluation(
        validation=validation,
        distance=distance,
        gap_percent=gap,
        crossings=crossings,
        canonical_routes=canonical_subroutes,
    )