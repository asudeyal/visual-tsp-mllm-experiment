"""Rastgele ve TSPLIB kaynaklı dinamik TSP problemlerini oluşturur."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from src.problem_instance import (
    ProblemInstance,
    ProblemSource,
    ReferenceSolution,
    ReferenceType,
)


SUPPORTED_TSPLIB_EDGE_WEIGHT_TYPES = frozenset(
    {
        "EUC_2D",
        "GEO",
    }
)

TSPLIB_GEO_EARTH_RADIUS_KM = 6378.388


def euc_2d_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> int:
    """TSPLIB EUC_2D kuralıyla iki koordinat arasındaki mesafeyi hesaplar."""

    return int(
        math.hypot(
            first[0] - second[0],
            first[1] - second[1],
        )
        + 0.5
    )


def geo_coordinate_to_radians(value: float) -> float:
    """TSPLIB DDD.MM koordinatını radyana dönüştürür."""

    degrees = int(value)
    minutes = value - degrees
    decimal_degrees = degrees + (5.0 * minutes / 3.0)
    return math.pi * decimal_degrees / 180.0


def geo_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> int:
    """TSPLIB GEO kuralıyla iki koordinat arasındaki mesafeyi hesaplar."""

    first_latitude = geo_coordinate_to_radians(first[0])
    first_longitude = geo_coordinate_to_radians(first[1])
    second_latitude = geo_coordinate_to_radians(second[0])
    second_longitude = geo_coordinate_to_radians(second[1])

    longitude_term = math.cos(
        first_longitude - second_longitude
    )
    latitude_difference_term = math.cos(
        first_latitude - second_latitude
    )
    latitude_sum_term = math.cos(
        first_latitude + second_latitude
    )
    acos_argument = 0.5 * (
        (1.0 + longitude_term) * latitude_difference_term
        - (1.0 - longitude_term) * latitude_sum_term
    )

    # Kayan nokta hataları acos aralığını çok küçük miktarda
    # aşabildiği için değer güvenli biçimde [-1, 1] aralığına alınır.
    bounded_argument = max(-1.0, min(1.0, acos_argument))

    return int(
        TSPLIB_GEO_EARTH_RADIUS_KM
        * math.acos(bounded_argument)
        + 1.0
    )


def tsplib_edge_distance(
    edge_weight_type: str,
    first: tuple[float, float],
    second: tuple[float, float],
) -> int:
    """Desteklenen TSPLIB türü için tek kenar mesafesini döndürür."""

    normalized_type = edge_weight_type.strip().upper()

    if normalized_type == "EUC_2D":
        return euc_2d_distance(first, second)

    if normalized_type == "GEO":
        return geo_distance(first, second)

    supported = ", ".join(
        sorted(SUPPORTED_TSPLIB_EDGE_WEIGHT_TYPES)
    )
    raise ValueError(
        f"EDGE_WEIGHT_TYPE={normalized_type or 'EMPTY'} "
        f"desteklenmiyor. Desteklenen türler: {supported}."
    )


def tsplib_route_distance(
    coordinates: dict[int, tuple[float, float]],
    route: tuple[int, ...] | list[int],
    *,
    edge_weight_type: str = "EUC_2D",
) -> int:
    """Kapalı bir TSPLIB rotasının toplam mesafesini hesaplar."""

    if len(route) < 3:
        raise ValueError(
            "Rota en az başlangıç, ziyaret ve dönüş içermelidir."
        )

    unknown = sorted(set(route) - set(coordinates))

    if unknown:
        raise ValueError(
            f"Rota bilinmeyen düğümler içeriyor: {unknown}"
        )

    return sum(
        tsplib_edge_distance(
            edge_weight_type,
            coordinates[first],
            coordinates[second],
        )
        for first, second in zip(route, route[1:])
    )


def _parse_metadata_line(
    line: str,
) -> tuple[str, str] | None:
    match = re.match(r"^([^:]+?)\s*:\s*(.+)$", line)

    if match is None:
        return None

    return (
        match.group(1).strip().upper(),
        match.group(2).strip(),
    )


def _parse_tsplib_coordinates(
    instance_file: Path,
) -> tuple[
    dict[str, str],
    dict[int, tuple[float, float]],
]:
    metadata: dict[str, str] = {}
    coordinates: dict[int, tuple[float, float]] = {}
    in_coordinates = False

    for raw_line in instance_file.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line == "NODE_COORD_SECTION":
            in_coordinates = True
            continue

        if line == "EOF":
            break

        if in_coordinates:
            if line.endswith("_SECTION"):
                break

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Geçersiz koordinat satırı: {line}"
                )

            node_id = int(parts[0])

            if node_id in coordinates:
                raise ValueError(
                    f"Düğüm {node_id} birden fazla kez tanımlanmış."
                )

            coordinates[node_id] = (
                float(parts[1]),
                float(parts[2]),
            )
            continue

        parsed = _parse_metadata_line(line)

        if parsed is not None:
            key, value = parsed
            metadata[key] = value

    if not coordinates:
        raise ValueError(
            "TSPLIB dosyasında NODE_COORD_SECTION bulunamadı."
        )

    return metadata, coordinates


def _parse_optimal_tour(
    tour_file: Path,
    *,
    expected_node_ids: set[int],
    depot_id: int,
) -> tuple[int, ...]:
    nodes: list[int] = []
    in_tour = False

    for raw_line in tour_file.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if line == "TOUR_SECTION":
            in_tour = True
            continue

        if not in_tour:
            continue

        if line in {"-1", "EOF"}:
            break

        nodes.extend(
            int(value)
            for value in line.split()
        )

    if not nodes:
        raise ValueError(
            "Optimal tur dosyasında TOUR_SECTION bulunamadı."
        )

    if len(nodes) != len(expected_node_ids):
        raise ValueError(
            "Optimal turdaki düğüm sayısı problemle uyuşmuyor."
        )

    if set(nodes) != expected_node_ids:
        missing = sorted(
            expected_node_ids - set(nodes)
        )
        unexpected = sorted(
            set(nodes) - expected_node_ids
        )

        raise ValueError(
            "Optimal tur problem düğümleriyle uyuşmuyor: "
            f"eksik={missing}, beklenmeyen={unexpected}"
        )

    if len(set(nodes)) != len(nodes):
        raise ValueError(
            "Optimal tur tekrarlanan düğüm içeriyor."
        )

    if depot_id not in nodes:
        raise ValueError(
            f"Optimal turda depo düğümü {depot_id} yok."
        )

    depot_index = nodes.index(depot_id)
    rotated = nodes[depot_index:] + nodes[:depot_index]

    return tuple([*rotated, depot_id])


def load_tsplib_problem(
    instance_file: Path,
    *,
    optimal_tour_file: Path | None = None,
    depot_id: int | None = None,
) -> ProblemInstance:
    """Koordinat tabanlı, desteklenen bir TSPLIB problemini yükler."""

    instance_file = Path(instance_file)

    if not instance_file.is_file():
        raise FileNotFoundError(
            f"TSPLIB problem dosyası bulunamadı: {instance_file}"
        )

    metadata, coordinates = _parse_tsplib_coordinates(
        instance_file
    )

    dimension = int(
        metadata.get("DIMENSION", "0")
    )

    if dimension != len(coordinates):
        raise ValueError(
            f"DIMENSION={dimension}, "
            f"okunan koordinat={len(coordinates)}."
        )

    edge_weight_type = metadata.get(
        "EDGE_WEIGHT_TYPE",
        "",
    ).upper()

    if edge_weight_type not in SUPPORTED_TSPLIB_EDGE_WEIGHT_TYPES:
        supported = ", ".join(
            sorted(SUPPORTED_TSPLIB_EDGE_WEIGHT_TYPES)
        )

        raise ValueError(
            f"EDGE_WEIGHT_TYPE="
            f"{edge_weight_type or 'EMPTY'} desteklenmiyor. "
            f"Desteklenen türler: {supported}."
        )

    selected_depot = (
        min(coordinates)
        if depot_id is None
        else int(depot_id)
    )

    if selected_depot not in coordinates:
        raise ValueError(
            f"Depo düğümü {selected_depot} problemde bulunamadı."
        )

    reference: ReferenceSolution | None = None
    normalized_tour_file: Path | None = None

    if optimal_tour_file is not None:
        normalized_tour_file = Path(
            optimal_tour_file
        )

        if not normalized_tour_file.is_file():
            raise FileNotFoundError(
                "Optimal tur dosyası bulunamadı: "
                f"{normalized_tour_file}"
            )

        route = _parse_optimal_tour(
            normalized_tour_file,
            expected_node_ids=set(coordinates),
            depot_id=selected_depot,
        )

        reference = ReferenceSolution(
            reference_type=(
                ReferenceType.TSPLIB_KNOWN_OPTIMUM
            ),
            distance=float(
                tsplib_route_distance(
                    coordinates,
                    route,
                    edge_weight_type=edge_weight_type,
                )
            ),
            is_proven_optimal=True,
            route=route,
            source_file=normalized_tour_file,
        )

    return ProblemInstance(
        name=metadata.get(
            "NAME",
            instance_file.stem,
        ),
        source_type=ProblemSource.TSPLIB,
        dimension=dimension,
        depot_id=selected_depot,
        edge_weight_type=edge_weight_type,
        coordinates=coordinates,
        source_file=instance_file,
        optimal_tour_file=normalized_tour_file,
        reference=reference,
    )


def generate_random_problem(
    num_nodes: int,
    *,
    seed: int = 42,
    low: float = 0.0,
    high: float = 5.0,
    depot_id: int = 0,
) -> ProblemInstance:
    """Belirtilen düğüm sayısıyla tekrar üretilebilir rastgele TSP oluşturur."""

    if num_nodes < 2:
        raise ValueError(
            "TSP için --num-nodes en az 2 olmalıdır."
        )

    if (
        not math.isfinite(low)
        or not math.isfinite(high)
        or high <= low
    ):
        raise ValueError(
            "Koordinat aralığında high, "
            "low değerinden büyük olmalıdır."
        )

    node_ids = list(range(num_nodes))

    if depot_id not in node_ids:
        raise ValueError(
            f"Depo düğümü 0..{num_nodes - 1} "
            "aralığında olmalıdır."
        )

    rng = np.random.default_rng(seed)

    points = rng.uniform(
        low,
        high,
        size=(num_nodes, 2),
    )

    coordinates = {
        node_id: (
            float(points[node_id][0]),
            float(points[node_id][1]),
        )
        for node_id in node_ids
    }

    return ProblemInstance(
        name=f"random_n{num_nodes}_seed{seed}",
        source_type=ProblemSource.RANDOM,
        dimension=num_nodes,
        depot_id=depot_id,
        edge_weight_type="EUC_2D_FLOAT",
        coordinates=coordinates,
        seed=seed,
    )
