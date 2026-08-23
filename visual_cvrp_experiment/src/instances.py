"""Tekrar üretilebilir CVRP araştırma örnekleri ve CVRPLIB yükleyicisi."""

from __future__ import annotations

import re
from pathlib import Path

from .problem import CVRPProblem, Node


CAPACITY_DEMO_10_NAME = "capacity_demo_10"


_CUSTOMER_DATA = (
    # node_id, x, y, demand
    (1, 15.0, 82.0, 2),
    (2, 48.0, 90.0, 3),
    (3, 84.0, 80.0, 1),
    (4, 12.0, 48.0, 3),
    (5, 88.0, 52.0, 2),
    (6, 18.0, 16.0, 1),
    (7, 50.0, 10.0, 2),
    (8, 82.0, 18.0, 3),
    (9, 55.0, 65.0, 1),
)


def build_capacity_demo_10() -> CVRPProblem:
    """İlk 10 düğümlü sabit CVRP örneğini oluştur."""

    return CVRPProblem(
        name=CAPACITY_DEMO_10_NAME,
        depot=Node(
            node_id=0,
            x=50.0,
            y=50.0,
            demand=0,
        ),
        customers=tuple(
            Node(
                node_id=node_id,
                x=x,
                y=y,
                demand=demand,
            )
            for node_id, x, y, demand
            in _CUSTOMER_DATA
        ),
        vehicle_capacity=6,
        vehicle_count=3,
    )


def _parse_cvrplib_sections(
    path: Path,
) -> dict[str, list[str]]:
    """CVRPLIB dosyasındaki ana bölümleri ayır."""

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    section_names = {
        "NODE_COORD_SECTION",
        "DEMAND_SECTION",
        "DEPOT_SECTION",
        "TOUR_SECTION",
        "EDGE_WEIGHT_SECTION",
    }

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        upper_line = line.upper()

        if upper_line in section_names:
            current_section = upper_line
            sections[current_section] = []
            continue

        if upper_line == "EOF":
            break

        if current_section is not None:
            sections[current_section].append(line)

    return sections


def _parse_cvrplib_metadata(
    path: Path,
) -> dict[str, str]:
    """CVRPLIB HEADER alanlarını oku."""

    metadata: dict[str, str] = {}

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip().upper()
        value = value.strip()

        metadata[key] = value

    return metadata


def _parse_vehicle_count(
    name: str,
    metadata: dict[str, str],
) -> int | None:
    """Problem adındaki -kN bilgisinden araç sayısını bul."""

    match = re.search(
        r"-k(\d+)$",
        name.strip(),
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    # Bazı veri setlerinde araç sayısı metadata içinde
    # ayrıca bulunabilir.
    for key in (
        "VEHICLES",
        "VEHICLE_COUNT",
        "NUMBER_OF_VEHICLES",
    ):
        value = metadata.get(key)

        if value is None:
            continue

        try:
            return int(value)
        except ValueError:
            continue

    return None


def build_cvrplib_problem(
    path: str | Path,
) -> CVRPProblem:
    """Standart CVRPLIB .vrp dosyasından CVRPProblem oluştur.

    Desteklenen temel CVRPLIB bölümleri:
    - NAME
    - DIMENSION
    - CAPACITY
    - NODE_COORD_SECTION
    - DEMAND_SECTION
    - DEPOT_SECTION

    Şimdilik koordinat tabanlı EUC_2D örnekleri hedeflenmektedir.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"CVRPLIB dosyası bulunamadı: {path}"
        )

    metadata = _parse_cvrplib_metadata(path)
    sections = _parse_cvrplib_sections(path)

    name = metadata.get(
        "NAME",
        path.stem,
    )

    edge_weight_type = metadata.get(
        "EDGE_WEIGHT_TYPE",
        "EUC_2D",
    ).upper()

    if edge_weight_type != "EUC_2D":
        raise ValueError(
            "Şu anda yalnızca EUC_2D CVRPLIB problemleri "
            f"destekleniyor. Bulunan tür: "
            f"{edge_weight_type}"
        )

    if "CAPACITY" not in metadata:
        raise ValueError(
            "CVRPLIB dosyasında CAPACITY bulunamadı."
        )

    try:
        vehicle_capacity = int(
            metadata["CAPACITY"]
        )
    except ValueError as error:
        raise ValueError(
            "CAPACITY tam sayı olmalıdır."
        ) from error

    if "NODE_COORD_SECTION" not in sections:
        raise ValueError(
            "CVRPLIB dosyasında NODE_COORD_SECTION bulunamadı."
        )

    if "DEMAND_SECTION" not in sections:
        raise ValueError(
            "CVRPLIB dosyasında DEMAND_SECTION bulunamadı."
        )

    if "DEPOT_SECTION" not in sections:
        raise ValueError(
            "CVRPLIB dosyasında DEPOT_SECTION bulunamadı."
        )

    coordinates: dict[int, tuple[float, float]] = {}

    for line in sections["NODE_COORD_SECTION"]:
        parts = line.split()

        if len(parts) < 3:
            continue

        node_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])

        coordinates[node_id] = (x, y)

    demands: dict[int, int] = {}

    for line in sections["DEMAND_SECTION"]:
        parts = line.split()

        if len(parts) < 2:
            continue

        node_id = int(parts[0])
        demand = int(parts[1])

        demands[node_id] = demand

    depot_ids: list[int] = []

    for line in sections["DEPOT_SECTION"]:
        value = int(line.split()[0])

        if value == -1:
            break

        depot_ids.append(value)

    if not depot_ids:
        raise ValueError(
            "CVRPLIB dosyasında depo bulunamadı."
        )

    if len(depot_ids) != 1:
        raise ValueError(
            "Bu proje şu anda yalnızca tek depolu CVRP "
            "problemlerini destekliyor."
        )

    depot_id = depot_ids[0]

    if depot_id not in coordinates:
        raise ValueError(
            f"Depo koordinatı bulunamadı: {depot_id}"
        )

    if depot_id not in demands:
        raise ValueError(
            f"Depo talebi bulunamadı: {depot_id}"
        )

    if demands[depot_id] != 0:
        raise ValueError(
            "CVRPLIB deposunun talebi 0 olmalıdır."
        )

    node_ids = sorted(coordinates)

    missing_demands = [
        node_id
        for node_id in node_ids
        if node_id not in demands
    ]

    if missing_demands:
        raise ValueError(
            "Talebi bulunmayan düğümler: "
            + ", ".join(
                str(node_id)
                for node_id in missing_demands
            )
        )

    customers = tuple(
        Node(
            node_id=node_id,
            x=coordinates[node_id][0],
            y=coordinates[node_id][1],
            demand=demands[node_id],
        )
        for node_id in node_ids
        if node_id != depot_id
    )

    vehicle_count = _parse_vehicle_count(
        name=name,
        metadata=metadata,
    )

    return CVRPProblem(
        name=name,
        depot=Node(
            node_id=depot_id,
            x=coordinates[depot_id][0],
            y=coordinates[depot_id][1],
            demand=0,
        ),
        customers=customers,
        vehicle_capacity=vehicle_capacity,
        vehicle_count=vehicle_count,
    )