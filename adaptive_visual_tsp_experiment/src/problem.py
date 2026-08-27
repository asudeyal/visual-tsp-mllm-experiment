"""TSPLIB EUC_2D loading and deterministic objective calculations."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .schemas import ProblemInstance


def _split_header(line: str) -> tuple[str, str] | None:
    if ":" in line:
        key, value = line.split(":", 1)
        return key.strip().upper(), value.strip()
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip().upper(), parts[1].strip()
    return None


def load_tsplib(
    path: str | Path,
    *,
    reference_optimum: float | None = None,
    strict_euc_2d: bool = True,
) -> ProblemInstance:
    path = Path(path)
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines()]

    headers: dict[str, str] = {}
    coordinates: dict[int, tuple[float, float]] = {}
    depot_nodes: list[int] = []
    section: str | None = None

    for line in lines:
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper == "NODE_COORD_SECTION":
            section = "coords"
            continue
        if upper == "DEPOT_SECTION":
            section = "depot"
            continue
        if upper in {"EOF", "DISPLAY_DATA_SECTION", "EDGE_WEIGHT_SECTION"}:
            section = None
            if upper == "EOF":
                break
            continue

        if section == "coords":
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Geçersiz NODE_COORD_SECTION satırı: {line}")
            node_id = int(parts[0])
            coordinates[node_id] = (float(parts[1]), float(parts[2]))
            continue

        if section == "depot":
            value = int(line.split()[0])
            if value == -1:
                section = None
            else:
                depot_nodes.append(value)
            continue

        header = _split_header(line)
        if header:
            headers[header[0]] = header[1]

    if not coordinates:
        raise ValueError("TSPLIB dosyasında NODE_COORD_SECTION bulunamadı")

    edge_weight_type = headers.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    if strict_euc_2d and edge_weight_type != "EUC_2D":
        raise ValueError(
            f"Bu deney EUC_2D ile sınırlandırılmıştır; dosya {edge_weight_type} kullanıyor"
        )

    dimension = int(headers.get("DIMENSION", len(coordinates)))
    if dimension != len(coordinates):
        raise ValueError(
            f"DIMENSION={dimension} fakat {len(coordinates)} koordinat okundu"
        )

    node_ids = tuple(sorted(coordinates))
    depot = depot_nodes[0] if depot_nodes else node_ids[0]
    if depot not in coordinates:
        raise ValueError(f"Depot node {depot} koordinatlarda yok")

    return ProblemInstance(
        name=headers.get("NAME", path.stem),
        dimension=dimension,
        node_ids=node_ids,
        coordinates=coordinates,
        depot=depot,
        edge_weight_type=edge_weight_type,
        source_path=str(path),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        reference_optimum=reference_optimum,
    )


def euc_2d_distance(problem: ProblemInstance, a: int, b: int) -> int:
    if problem.edge_weight_type != "EUC_2D":
        raise ValueError("euc_2d_distance yalnız EUC_2D için kullanılabilir")
    ax, ay = problem.coordinates[a]
    bx, by = problem.coordinates[b]
    return int(math.hypot(ax - bx, ay - by) + 0.5)


def route_length(problem: ProblemInstance, route: tuple[int, ...] | list[int]) -> float:
    if len(route) < 2:
        return 0.0
    return float(
        sum(euc_2d_distance(problem, a, b) for a, b in zip(route, route[1:]))
    )
