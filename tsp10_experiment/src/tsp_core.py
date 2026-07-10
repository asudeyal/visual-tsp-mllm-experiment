"""10 noktalı görsel TSP deneyi için ortak ve test edilebilir fonksiyonlar.

Bu modül henüz bir LLM çağrısı yapmaz. Amacı, makaledeki deneyin bütün
yöntemlerinde aynı kalması gereken veri, rota doğrulama ve mesafe hesaplama
adımlarını tek yerde toplamaktır.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Sequence

import matplotlib

# Grafikleri ekranda açmadan doğrudan PNG dosyasına kaydeder.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


Point = tuple[float, float]


@dataclass(frozen=True)
class RouteValidation:
    """Bir TSP rotasının yapısal olarak geçerli olup olmadığını açıklar."""

    is_valid: bool
    starts_at_depot: bool
    ends_at_depot: bool
    missing_nodes: list[int]
    repeated_nodes: list[int]
    unexpected_nodes: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TSPSolution:
    """Bir çözücünün döndürdüğü rota ve gerçek Öklid mesafesi."""

    method: str
    route: list[int]
    distance: float
    validation: RouteValidation

    def to_dict(self) -> dict:
        result = asdict(self)
        result["validation"] = self.validation.to_dict()
        return result


def generate_locations(
    num_locations: int = 10,
    seed: int = 42,
    low: float = 0.0,
    high: float = 5.0,
) -> list[Point]:
    """Makalede olduğu gibi 5x5 alanda uniform dağılımlı noktalar üretir.

    Sıfır numaralı nokta depodur. ``seed`` kullanılması aynı deneyin yeniden
    üretilebilmesini sağlar; orijinal notebook'ta bu güvence yoktur.
    """

    if num_locations < 2:
        raise ValueError("TSP için depo dahil en az iki nokta gerekir.")

    rng = np.random.default_rng(seed)
    points = rng.uniform(low, high, size=(num_locations, 2))
    return [(float(x), float(y)) for x, y in points]


def euclidean_distance_matrix(locations: Sequence[Point]) -> np.ndarray:
    """Bütün nokta çiftleri arasındaki Öklid uzaklıklarını hesaplar."""

    points = np.asarray(locations, dtype=float)
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def route_distance(locations: Sequence[Point], route: Sequence[int]) -> float:
    """Verilen sıradaki kapalı veya açık rotanın toplam gerçek mesafesi."""

    distances = euclidean_distance_matrix(locations)
    return float(sum(distances[a, b] for a, b in zip(route, route[1:])))


def validate_tsp_route(
    route: Sequence[int], num_locations: int, depot: int = 0
) -> RouteValidation:
    """Rotanın depodan başlayıp bütün düğümleri tam bir kez gezdiğini denetler."""

    route_list = [int(node) for node in route]
    starts_at_depot = bool(route_list) and route_list[0] == depot
    ends_at_depot = bool(route_list) and route_list[-1] == depot
    interior = route_list[1:-1] if len(route_list) >= 2 else []

    expected = set(range(num_locations)) - {depot}
    interior_expected = [node for node in interior if node in expected]
    visited = set(interior_expected)
    missing_nodes = sorted(expected - visited)
    repeated_nodes = sorted(
        node for node in expected if interior_expected.count(node) > 1
    )
    unexpected_nodes = sorted(
        set(node for node in interior if node not in expected)
    )

    is_valid = (
        starts_at_depot
        and ends_at_depot
        and not missing_nodes
        and not repeated_nodes
        and not unexpected_nodes
        and len(interior) == num_locations - 1
    )

    return RouteValidation(
        is_valid=is_valid,
        starts_at_depot=starts_at_depot,
        ends_at_depot=ends_at_depot,
        missing_nodes=missing_nodes,
        repeated_nodes=repeated_nodes,
        unexpected_nodes=unexpected_nodes,
    )


def solve_ortools_tsp(
    locations: Sequence[Point], time_limit_seconds: int = 2
) -> TSPSolution:
    """Makalede kullanılan OR-Tools ayarlarıyla tek araçlı TSP'yi çözer.

    Makale son deneylerde 120 saniye kullanır. Geliştirme sırasında 10 nokta
    için kısa bir süre yeterlidir; final deneyinde bu parametre 120 yapılabilir.
    """

    num_locations = len(locations)
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    distances = euclidean_distance_matrix(locations)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distances[from_node, to_node] * 1000)

    callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(callback_index)

    # Orijinal notebook'taki Distance boyutu ve span katsayısı korunmuştur.
    routing.AddDimension(callback_index, 0, 300_000, True, "Distance")
    routing.GetDimensionOrDie("Distance").SetGlobalSpanCostCoefficient(100)

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    )
    search.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search.time_limit.seconds = int(time_limit_seconds)

    assignment = routing.SolveWithParameters(search)
    if assignment is None:
        raise RuntimeError("OR-Tools geçerli bir TSP çözümü bulamadı.")

    route: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = assignment.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))

    validation = validate_tsp_route(route, num_locations)
    return TSPSolution(
        method="or_tools",
        route=route,
        distance=route_distance(locations, route),
        validation=validation,
    )


def solve_exact_tsp(locations: Sequence[Point]) -> TSPSolution:
    """10 noktalı deney için bütün sıralamaları deneyerek kesin optimumu bulur.

    Dokuz ziyaret noktası için 9! = 362.880 olasılık vardır. Bu yöntem eğitim
    ve doğrulama amacıyla 10 nokta için uygundur; büyük TSP'lerde kullanılmaz.
    """

    num_locations = len(locations)
    if num_locations > 10:
        raise ValueError("Kesin brute-force çözücü en fazla 10 nokta içindir.")

    distances = euclidean_distance_matrix(locations)
    best_distance = float("inf")
    best_route: list[int] | None = None

    for order in permutations(range(1, num_locations)):
        candidate = (0, *order, 0)
        candidate_distance = sum(
            distances[a, b] for a, b in zip(candidate, candidate[1:])
        )
        if candidate_distance < best_distance:
            best_distance = float(candidate_distance)
            best_route = list(candidate)

    if best_route is None:
        raise RuntimeError("Kesin TSP çözümü oluşturulamadı.")

    validation = validate_tsp_route(best_route, num_locations)
    return TSPSolution(
        method="exact_brute_force",
        route=best_route,
        distance=best_distance,
        validation=validation,
    )


def percentage_gap(candidate_distance: float, reference_distance: float) -> float:
    """Aday çözümün referansa göre yüzde farkını hesaplar."""

    if reference_distance <= 0:
        raise ValueError("Referans mesafesi sıfırdan büyük olmalıdır.")
    return 100.0 * (candidate_distance - reference_distance) / reference_distance


def plot_problem(locations: Sequence[Point], output_path: Path) -> None:
    """LLM'ye verilecek, yalnızca noktaları ve düğüm numaralarını içeren görsel."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    for node, (x, y) in enumerate(locations):
        if node == 0:
            ax.plot(x, y, "ks", markersize=11, label="Depot (0)")
        else:
            ax.plot(x, y, "bo", markersize=7)
        ax.annotate(str(node), (x, y), xytext=(6, 6), textcoords="offset points")

    ax.set_xlim(-0.25, 5.25)
    ax.set_ylim(-0.25, 5.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("10-Node TSP")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_solution(
    locations: Sequence[Point], solution: TSPSolution, output_path: Path
) -> None:
    """Bir TSP çözümünü makaledeki rotalara benzer biçimde görselleştirir."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(locations, dtype=float)
    route_points = points[solution.route]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        route_points[:, 0],
        route_points[:, 1],
        color="tab:blue",
        marker="o",
        linewidth=2,
    )
    for node, (x, y) in enumerate(locations):
        marker = "s" if node == 0 else "o"
        color = "black" if node == 0 else "tab:red"
        ax.plot(x, y, marker=marker, color=color, markersize=9)
        ax.annotate(str(node), (x, y), xytext=(6, 6), textcoords="offset points")

    ax.set_xlim(-0.25, 5.25)
    ax.set_ylim(-0.25, 5.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{solution.method} | distance={solution.distance:.4f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
