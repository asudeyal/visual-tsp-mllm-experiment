"""CVRPLIB loading and deterministic observer-side CVRP objective functions."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

from .schemas import ProblemInstance


def _split_header(
    line: str,
) -> tuple[str, str] | None:

    if ":" in line:

        key, value = line.split(
            ":",
            1,
        )

        return (
            key.strip().upper(),
            value.strip(),
        )

    parts = line.split(
        maxsplit=1
    )

    if len(parts) == 2:

        return (
            parts[0].strip().upper(),
            parts[1].strip(),
        )

    return None


def _validated_max_vehicles(
    max_vehicles: int | None,
) -> int | None:

    if max_vehicles is None:
        return None

    if (
        isinstance(max_vehicles, bool)
        or not isinstance(max_vehicles, int)
    ):
        raise ValueError(
            "max_vehicles bir pozitif integer veya None "
            "olmalıdır"
        )

    if max_vehicles < 1:
        raise ValueError(
            "max_vehicles en az 1 olmalıdır"
        )

    return max_vehicles


def load_cvrplib(
    path: str | Path,
    *,
    reference_optimum: float | None = None,
    strict_euc_2d: bool = True,
    max_vehicles: int | None = None,
) -> ProblemInstance:
    """Load a single-depot CVRPLIB instance.

    Capacity, demands, the optional vehicle limit, and
    reference optimum are observer-side data.
    They are never passed to the model as text.
    """

    path = Path(path)

    raw_bytes = path.read_bytes()

    text = raw_bytes.decode(
        "utf-8",
        errors="replace",
    )

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    headers: dict[str, str] = {}

    coordinates: dict[
        int,
        tuple[float, float],
    ] = {}

    demands: dict[int, int] = {}

    depot_nodes: list[int] = []

    section: str | None = None

    for line in lines:

        if not line or line.startswith("#"):
            continue

        upper = line.upper()

        if upper == "NODE_COORD_SECTION":

            section = "coords"
            continue

        if upper == "DEMAND_SECTION":

            section = "demand"
            continue

        if upper == "DEPOT_SECTION":

            section = "depot"
            continue

        if upper in {
            "EOF",
            "DISPLAY_DATA_SECTION",
            "EDGE_WEIGHT_SECTION",
        }:

            section = None

            if upper == "EOF":
                break

            continue

        if section == "coords":

            parts = line.split()

            if len(parts) < 3:
                raise ValueError(
                    "Geçersiz NODE_COORD_SECTION "
                    f"satırı: {line}"
                )

            node_id = int(parts[0])

            coordinates[node_id] = (
                float(parts[1]),
                float(parts[2]),
            )

            continue

        if section == "demand":

            parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    "Geçersiz DEMAND_SECTION "
                    f"satırı: {line}"
                )

            node_id = int(parts[0])

            demands[node_id] = int(parts[1])

            continue

        if section == "depot":

            value = int(
                line.split()[0]
            )

            if value == -1:

                section = None

            else:

                depot_nodes.append(value)

            continue

        header = _split_header(line)

        if header is not None:

            key, value = header

            headers[key] = value

    if not coordinates:

        raise ValueError(
            "CVRPLIB dosyasında "
            "NODE_COORD_SECTION bulunamadı"
        )

    problem_type = headers.get(
        "TYPE",
        "CVRP",
    ).upper()

    if problem_type != "CVRP":

        raise ValueError(
            "Bu yükleyici yalnız CVRP örneklerini "
            "kabul eder; "
            f"dosya türü: {problem_type}"
        )

    edge_weight_type = headers.get(
        "EDGE_WEIGHT_TYPE",
        "EUC_2D",
    ).upper()

    if (
        strict_euc_2d
        and edge_weight_type != "EUC_2D"
    ):

        raise ValueError(
            "Bu deney EUC_2D ile sınırlandırılmıştır; "
            f"dosya {edge_weight_type} kullanıyor"
        )

    dimension = int(
        headers.get(
            "DIMENSION",
            len(coordinates),
        )
    )

    if dimension != len(coordinates):

        raise ValueError(
            f"DIMENSION={dimension} fakat "
            f"{len(coordinates)} koordinat okundu"
        )

    capacity = int(
        headers.get(
            "CAPACITY",
            0,
        )
    )

    if capacity <= 0:

        raise ValueError(
            "CVRPLIB dosyasında geçerli "
            "CAPACITY tanımı bulunamadı"
        )

    node_ids = tuple(
        sorted(coordinates)
    )

    if depot_nodes:

        unique_depots = tuple(
            sorted(
                set(depot_nodes)
            )
        )

        if len(unique_depots) != 1:

            raise ValueError(
                "Bu deney yalnız tek depolu "
                "CVRP örneklerini destekler"
            )

        depot = unique_depots[0]

    else:

        depot = node_ids[0]

    if depot not in coordinates:

        raise ValueError(
            f"Depot node {depot} "
            "koordinatlarda yok"
        )

    missing_demand_nodes = sorted(
        set(node_ids)
        - set(demands)
    )

    if missing_demand_nodes:

        raise ValueError(
            "DEMAND_SECTION şu düğümler için "
            "talep içermiyor: "
            f"{missing_demand_nodes}"
        )

    unknown_demand_nodes = sorted(
        set(demands)
        - set(node_ids)
    )

    if unknown_demand_nodes:

        raise ValueError(
            "DEMAND_SECTION koordinatı olmayan "
            "düğümler içeriyor: "
            f"{unknown_demand_nodes}"
        )

    negative_demand_nodes = sorted(
        node
        for node, demand in demands.items()
        if demand < 0
    )

    if negative_demand_nodes:

        raise ValueError(
            "Negatif müşteri talebi kabul edilmez: "
            f"{negative_demand_nodes}"
        )

    if demands[depot] != 0:

        raise ValueError(
            f"Depot node {depot} için talep 0 "
            "olmalıdır; "
            f"okunan değer: {demands[depot]}"
        )

    oversized_customers = sorted(
        node
        for node in node_ids
        if (
            node != depot
            and demands[node] > capacity
        )
    )

    if oversized_customers:

        raise ValueError(
            "Tek başına araç kapasitesini aşan "
            "müşteriler var: "
            f"{oversized_customers}"
        )

    if reference_optimum is not None:

        reference_optimum = float(
            reference_optimum
        )

        if reference_optimum <= 0:

            raise ValueError(
                "reference_optimum pozitif "
                "olmalıdır"
            )

    return ProblemInstance(
        name=headers.get(
            "NAME",
            path.stem,
        ),
        dimension=dimension,
        node_ids=node_ids,
        coordinates=coordinates,
        depot=depot,
        capacity=capacity,
        demands=demands,
        max_vehicles=_validated_max_vehicles(
            max_vehicles
        ),
        edge_weight_type=edge_weight_type,
        source_path=str(path),
        source_sha256=hashlib.sha256(
            raw_bytes
        ).hexdigest(),
        reference_optimum=reference_optimum,
    )


def euc_2d_distance(
    problem: ProblemInstance,
    a: int,
    b: int,
) -> int:
    """Return TSPLIB-style rounded Euclidean distance."""

    if problem.edge_weight_type != "EUC_2D":

        raise ValueError(
            "euc_2d_distance yalnız EUC_2D "
            "için kullanılabilir"
        )

    ax, ay = problem.coordinates[a]

    bx, by = problem.coordinates[b]

    return int(
        math.hypot(
            ax - bx,
            ay - by,
        )
        + 0.5
    )


def route_length(
    problem: ProblemInstance,
    route: Iterable[int],
) -> float:
    """Calculate the distance of one vehicle route."""

    route_tuple = tuple(route)

    if len(route_tuple) < 2:
        return 0.0

    return float(
        sum(
            euc_2d_distance(
                problem,
                first,
                second,
            )
            for first, second in zip(
                route_tuple,
                route_tuple[1:],
            )
        )
    )


def multi_route_length(
    problem: ProblemInstance,
    routes: Iterable[Iterable[int]],
) -> float:
    """Calculate the total distance across all vehicle routes."""

    return sum(
        route_length(
            problem,
            route,
        )
        for route in routes
    )