"""Dinamik TSP doğrulama, çözme, görselleştirme ve dosya araçları."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.problem_instance import ProblemInstance
from src.problem_loader import load_tsplib_problem


# Geçiş sürecinde eski importları bozmamak için korunur.
TSPLIBInstance = ProblemInstance
DEPOT_ID = 1
KNOWN_OPTIMUM = 426


@dataclass(frozen=True)
class RouteValidation:
    is_valid: bool
    starts_at_depot: bool
    ends_at_depot: bool
    missing_nodes: list[int]
    repeated_nodes: list[int]
    unexpected_nodes: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_tsplib(path: Path) -> ProblemInstance:
    """Eski çağrılar için genel TSPLIB yükleyiciye geçiş sarmalayıcısı."""

    return load_tsplib_problem(Path(path))


def parse_tsplib_tour(path: Path, *, depot_id: int = 1) -> list[int]:
    """Eski çağrılar için bir TOUR_SECTION dosyasını kapalı tura dönüştürür."""

    nodes: list[int] = []
    in_tour = False
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "TOUR_SECTION":
            in_tour = True
            continue
        if not in_tour:
            continue
        if line in {"-1", "EOF"}:
            break
        nodes.extend(int(value) for value in line.split())

    if not nodes:
        raise ValueError("Tur dosyasında TOUR_SECTION bulunamadı.")
    if depot_id not in nodes:
        raise ValueError(f"Turda depo düğümü {depot_id} bulunamadı.")

    depot_index = nodes.index(depot_id)
    rotated = nodes[depot_index:] + nodes[:depot_index]
    return [*rotated, depot_id]


def euc_2d_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> int:
    """TSPLIB EUC_2D mesafesini geriye uyumlu olarak hesaplar."""

    return int(
        math.hypot(
            first[0] - second[0],
            first[1] - second[1],
        )
        + 0.5
    )


def _edge_distance(
    instance: ProblemInstance,
    first: int,
    second: int,
) -> float:
    first_point = instance.coordinates[first]
    second_point = instance.coordinates[second]
    distance = math.hypot(
        first_point[0] - second_point[0],
        first_point[1] - second_point[1],
    )
    if instance.edge_weight_type == "EUC_2D":
        return float(int(distance + 0.5))
    if instance.edge_weight_type == "EUC_2D_FLOAT":
        return distance
    raise ValueError(
        f"Desteklenmeyen mesafe türü: {instance.edge_weight_type}"
    )


def distance_matrix(instance: ProblemInstance) -> list[list[float]]:
    ids = instance.node_ids
    return [
        [_edge_distance(instance, first, second) for second in ids]
        for first in ids
    ]


def route_distance(
    instance: ProblemInstance,
    route: list[int] | tuple[int, ...],
) -> float | int:
    if len(route) < 2:
        raise ValueError("Mesafe için rota en az iki düğüm içermelidir.")

    legal = set(instance.node_ids)
    unknown = sorted(set(route) - legal)
    if unknown:
        raise ValueError(f"Rota bilinmeyen düğümler içeriyor: {unknown}")

    distance = sum(
        _edge_distance(instance, first, second)
        for first, second in zip(route, route[1:])
    )
    if instance.edge_weight_type == "EUC_2D":
        return int(distance)
    return distance


def validate_route(
    route: list[int] | tuple[int, ...],
    instance: ProblemInstance,
    *,
    depot_id: int | None = None,
) -> RouteValidation:
    selected_depot = instance.depot_id if depot_id is None else depot_id
    expected = set(instance.node_ids)
    starts = bool(route) and route[0] == selected_depot
    ends = bool(route) and route[-1] == selected_depot
    visits = list(route[:-1] if ends else route)
    counts = Counter(visits)
    missing = sorted(expected - set(visits))
    repeated = sorted(
        node for node, count in counts.items() if count > 1
    )
    unexpected = sorted(set(visits) - expected)
    valid = (
        starts
        and ends
        and not missing
        and not repeated
        and not unexpected
        and len(route) == instance.dimension + 1
    )
    return RouteValidation(
        is_valid=valid,
        starts_at_depot=starts,
        ends_at_depot=ends,
        missing_nodes=missing,
        repeated_nodes=repeated,
        unexpected_nodes=unexpected,
    )


def percentage_gap(distance: float, reference: float) -> float:
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Gap referansı pozitif ve sonlu olmalıdır.")
    return 100.0 * (distance - reference) / reference


def evaluate_route(
    instance: ProblemInstance,
    route: list[int] | tuple[int, ...],
    *,
    reference_distance: float | None = None,
) -> dict[str, Any]:
    normalized_route = [int(node) for node in route]
    validation = validate_route(normalized_route, instance)
    legal_ids = all(
        node in instance.coordinates for node in normalized_route
    )
    distance = (
        route_distance(instance, normalized_route)
        if legal_ids and len(normalized_route) >= 2
        else None
    )
    reference = (
        reference_distance
        if reference_distance is not None
        else (
            instance.reference.distance
            if instance.reference is not None
            else None
        )
    )
    gap = (
        percentage_gap(distance, reference)
        if (
            distance is not None
            and reference is not None
            and validation.is_valid
        )
        else None
    )
    return {
        "route": normalized_route,
        "validation": validation.to_dict(),
        "legal_node_ids": legal_ids,
        "distance": distance,
        "reference_distance": reference,
        "gap_to_reference_percent": gap,
        # Eski EIL51 sonuç okuyucuları için geçici uyumluluk alanı.
        "gap_to_known_optimum_percent": gap,
    }


def solve_ortools(
    instance: ProblemInstance,
    *,
    time_limit_seconds: int = 30,
    depot_id: int | None = None,
) -> dict[str, Any]:
    """Problemi OR-Tools SAVINGS + GUIDED_LOCAL_SEARCH ile çözer."""

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    selected_depot = (
        instance.depot_id if depot_id is None else int(depot_id)
    )
    if selected_depot not in instance.coordinates:
        raise ValueError(f"Depo düğümü {selected_depot} problemde yok.")
    if time_limit_seconds < 1:
        raise ValueError("OR-Tools süre sınırı en az 1 saniye olmalıdır.")

    ids = instance.node_ids
    index_by_id = {
        node_id: index for index, node_id in enumerate(ids)
    }
    raw_matrix = distance_matrix(instance)
    scale = 1 if instance.edge_weight_type == "EUC_2D" else 1_000_000
    cost_matrix = [
        [int(round(value * scale)) for value in row]
        for row in raw_matrix
    ]

    manager = pywrapcp.RoutingIndexManager(
        len(ids),
        1,
        index_by_id[selected_depot],
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(
        from_index: int,
        to_index: int,
    ) -> int:
        return cost_matrix[
            manager.IndexToNode(from_index)
        ][manager.IndexToNode(to_index)]

    callback_index = routing.RegisterTransitCallback(
        distance_callback
    )
    routing.SetArcCostEvaluatorOfAllVehicles(callback_index)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.seconds = int(time_limit_seconds)

    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError(
            f"OR-Tools {instance.name} için çözüm üretemedi."
        )

    route: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(ids[manager.IndexToNode(index)])
        index = solution.Value(routing.NextVar(index))
    route.append(ids[manager.IndexToNode(index)])

    return {
        "method": "or_tools_savings_guided_local_search",
        **evaluate_route(instance, route),
    }


def _plot_nodes(
    instance: ProblemInstance,
    ax: Any,
) -> None:
    for node_id in instance.node_ids:
        x, y = instance.coordinates[node_id]
        if node_id == instance.depot_id:
            ax.scatter(
                x,
                y,
                marker="s",
                s=120,
                color="black",
                zorder=4,
            )
        else:
            ax.scatter(
                x,
                y,
                s=45,
                color="#2b6cb0",
                zorder=3,
            )
        ax.annotate(
            str(node_id),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            zorder=5,
        )


def plot_problem(
    instance: ProblemInstance,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_nodes(instance, ax)
    ax.set_title(
        f"{instance.name} — {instance.dimension} nodes — "
        f"depot {instance.depot_id}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_route(
    instance: ProblemInstance,
    route: list[int] | tuple[int, ...],
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    legal_pairs = [
        (first, second)
        for first, second in zip(route, route[1:])
        if (
            first in instance.coordinates
            and second in instance.coordinates
        )
    ]
    for first, second in legal_pairs:
        x1, y1 = instance.coordinates[first]
        x2, y2 = instance.coordinates[second]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#d9485f",
            linewidth=1.4,
            zorder=1,
        )
    _plot_nodes(instance, ax)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def normalize_run_id(run_id: str | None) -> str:
    value = run_id or "default"
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}",
        value,
    ):
        raise ValueError(
            "run-id geçersiz; harf/rakamla başlamalıdır."
        )
    return value


def method_dir(
    output_dir: Path,
    run_id: str | None,
    method: str,
) -> Path:
    return (
        Path(output_dir)
        / "runs"
        / normalize_run_id(run_id)
        / method
    )
