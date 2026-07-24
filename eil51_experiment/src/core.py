"""TSPLIB eil51 okuma, doğrulama, çözme ve görselleştirme araçları."""

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


DEPOT_ID = 1
KNOWN_OPTIMUM = 426


@dataclass(frozen=True)
class TSPLIBInstance:
    name: str
    dimension: int
    edge_weight_type: str
    coordinates: dict[int, tuple[float, float]]

    @property
    def node_ids(self) -> list[int]:
        return sorted(self.coordinates)


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


def parse_tsplib(path: Path) -> TSPLIBInstance:
    """NODE_COORD_SECTION içeren EUC_2D TSPLIB dosyasını okur."""

    metadata: dict[str, str] = {}
    coordinates: dict[int, tuple[float, float]] = {}
    in_coordinates = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "NODE_COORD_SECTION":
            in_coordinates = True
            continue
        if line == "EOF":
            break
        if in_coordinates:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Geçersiz koordinat satırı: {line}")
            node_id = int(parts[0])
            coordinates[node_id] = (float(parts[1]), float(parts[2]))
            continue
        match = re.match(r"^([^:]+)\s*:\s*(.+)$", line)
        if match:
            metadata[match.group(1).strip()] = match.group(2).strip()

    dimension = int(metadata.get("DIMENSION", "0"))
    edge_weight_type = metadata.get("EDGE_WEIGHT_TYPE", "")
    if dimension != len(coordinates):
        raise ValueError(
            f"DIMENSION={dimension}, okunan koordinat={len(coordinates)}."
        )
    if edge_weight_type != "EUC_2D":
        raise ValueError("Bu deney yalnız TSPLIB EUC_2D örneklerini destekler.")
    if sorted(coordinates) != list(range(1, dimension + 1)):
        raise ValueError("Düğüm kimlikleri 1..DIMENSION aralığında olmalıdır.")
    return TSPLIBInstance(
        name=metadata.get("NAME", path.stem),
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        coordinates=coordinates,
    )


def parse_tsplib_tour(path: Path, *, depot_id: int = DEPOT_ID) -> list[int]:
    """TSPLIB TOUR_SECTION dosyasını depoda başlayıp bitecek biçime getirir."""

    nodes: list[int] = []
    in_tour = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "TOUR_SECTION":
            in_tour = True
            continue
        if not in_tour:
            continue
        if line in {"-1", "EOF"}:
            break
        nodes.extend(int(value) for value in line.split())
    if depot_id not in nodes:
        raise ValueError(f"Optimum turda depo düğümü {depot_id} bulunamadı.")
    depot_index = nodes.index(depot_id)
    rotated = nodes[depot_index:] + nodes[:depot_index]
    return rotated + [depot_id]


def euc_2d_distance(
    first: tuple[float, float], second: tuple[float, float]
) -> int:
    """TSPLIB EUC_2D kuralı: Öklid mesafesini en yakın tam sayıya yuvarlar."""

    return int(math.hypot(first[0] - second[0], first[1] - second[1]) + 0.5)


def distance_matrix(instance: TSPLIBInstance) -> list[list[int]]:
    ids = instance.node_ids
    return [
        [euc_2d_distance(instance.coordinates[a], instance.coordinates[b]) for b in ids]
        for a in ids
    ]


def route_distance(instance: TSPLIBInstance, route: list[int]) -> int:
    if len(route) < 2:
        raise ValueError("Mesafe için rota en az iki düğüm içermelidir.")
    legal = set(instance.node_ids)
    if any(node not in legal for node in route):
        raise ValueError("Rota, eil51 dışında bir düğüm içeriyor.")
    return sum(
        euc_2d_distance(instance.coordinates[a], instance.coordinates[b])
        for a, b in zip(route, route[1:])
    )


def validate_route(
    route: list[int],
    instance: TSPLIBInstance,
    *,
    depot_id: int = DEPOT_ID,
) -> RouteValidation:
    expected = set(instance.node_ids)
    starts = bool(route) and route[0] == depot_id
    ends = bool(route) and route[-1] == depot_id
    visits = route[:-1] if ends else route
    counts = Counter(visits)
    missing = sorted(expected - set(visits))
    repeated = sorted(node for node, count in counts.items() if count > 1)
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


def percentage_gap(distance: float, reference: float = KNOWN_OPTIMUM) -> float:
    return 100.0 * (distance - reference) / reference


def evaluate_route(instance: TSPLIBInstance, route: list[int]) -> dict[str, Any]:
    validation = validate_route(route, instance)
    legal_ids = all(node in instance.coordinates for node in route)
    distance = route_distance(instance, route) if legal_ids and len(route) >= 2 else None
    return {
        "route": route,
        "validation": validation.to_dict(),
        "legal_node_ids": legal_ids,
        "distance": distance,
        "gap_to_known_optimum_percent": (
            percentage_gap(distance)
            if distance is not None and validation.is_valid
            else None
        ),
    }


def solve_ortools(
    instance: TSPLIBInstance,
    *,
    time_limit_seconds: int = 30,
    depot_id: int = DEPOT_ID,
) -> dict[str, Any]:
    """Eil51'i OR-Tools SAVINGS + GUIDED_LOCAL_SEARCH ile çözer."""

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    ids = instance.node_ids
    index_by_id = {node_id: index for index, node_id in enumerate(ids)}
    matrix = distance_matrix(instance)
    manager = pywrapcp.RoutingIndexManager(
        len(ids), 1, index_by_id[depot_id]
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    callback_index = routing.RegisterTransitCallback(distance_callback)
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
        raise RuntimeError("OR-Tools eil51 için çözüm üretemedi.")

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


def _plot_nodes(instance: TSPLIBInstance, ax: Any) -> None:
    for node_id in instance.node_ids:
        x, y = instance.coordinates[node_id]
        if node_id == DEPOT_ID:
            ax.scatter(x, y, marker="s", s=120, color="black", zorder=4)
        else:
            ax.scatter(x, y, s=45, color="#2b6cb0", zorder=3)
        ax.annotate(
            str(node_id),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            zorder=5,
        )


def plot_problem(instance: TSPLIBInstance, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_nodes(instance, ax)
    ax.set_title("TSPLIB eil51 — Node 1 is the depot")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_route(
    instance: TSPLIBInstance,
    route: list[int],
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    legal_pairs = [
        (a, b)
        for a, b in zip(route, route[1:])
        if a in instance.coordinates and b in instance.coordinates
    ]
    for a, b in legal_pairs:
        x1, y1 = instance.coordinates[a]
        x2, y2 = instance.coordinates[b]
        ax.plot([x1, x2], [y1, y2], color="#d9485f", linewidth=1.4, zorder=1)
    _plot_nodes(instance, ax)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def normalize_run_id(run_id: str | None) -> str:
    value = run_id or "default"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", value):
        raise ValueError("run-id geçersiz; harf/rakamla başlamalıdır.")
    return value


def method_dir(output_dir: Path, run_id: str | None, method: str) -> Path:
    return output_dir / "runs" / normalize_run_id(run_id) / method
