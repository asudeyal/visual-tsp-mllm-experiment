"""Dinamik TSP deneylerinin ortak problem ve referans veri modelleri."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


Point = tuple[float, float]


class ProblemSource(str, Enum):
    """Bir TSP probleminin hangi kaynaktan oluşturulduğunu belirtir."""

    RANDOM = "random"
    TSPLIB = "tsplib"


class ReferenceType(str, Enum):
    """Karşılaştırma çözümünün kanıt düzeyini ve kaynağını belirtir."""

    TSPLIB_KNOWN_OPTIMUM = "tsplib_known_optimum"
    EXACT_BRUTE_FORCE = "exact_brute_force"
    OR_TOOLS_HEURISTIC = "or_tools_heuristic"


@dataclass(frozen=True)
class ReferenceSolution:
    """Bir problem için bilinen veya hesaplanan karşılaştırma çözümü."""

    reference_type: ReferenceType
    distance: float
    is_proven_optimal: bool
    route: tuple[int, ...] | None = None
    source_file: Path | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.distance) or self.distance <= 0:
            raise ValueError("Referans mesafesi pozitif ve sonlu olmalıdır.")
        if self.route is not None and len(self.route) < 3:
            raise ValueError("Referans rota kapalı bir TSP turu olmalıdır.")

    def to_dict(self, *, include_route: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.reference_type.value,
            "distance": self.distance,
            "is_proven_optimal": self.is_proven_optimal,
            "source_file": (
                self.source_file.as_posix()
                if self.source_file is not None
                else None
            ),
        }

        if include_route:
            result["route"] = (
                list(self.route)
                if self.route is not None
                else None
            )

        return result


@dataclass(frozen=True)
class ProblemInstance:
    """Rastgele veya TSPLIB kaynaklı tek satıcılı TSP problemi."""

    name: str
    source_type: ProblemSource
    dimension: int
    depot_id: int
    edge_weight_type: str
    coordinates: dict[int, Point]
    seed: int | None = None
    source_file: Path | None = None
    optimal_tour_file: Path | None = None
    reference: ReferenceSolution | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Problem adı boş olamaz.")

        if self.dimension < 2:
            raise ValueError("TSP için en az iki düğüm gerekir.")

        if len(self.coordinates) != self.dimension:
            raise ValueError(
                "Problem dimension değeri ile koordinat sayısı uyuşmuyor."
            )

        if self.depot_id not in self.coordinates:
            raise ValueError(
                f"Depo düğümü {self.depot_id} koordinatlarda yok."
            )

        if not self.edge_weight_type.strip():
            raise ValueError("EDGE_WEIGHT_TYPE boş olamaz.")

        normalized: dict[int, Point] = {}

        for node_id, point in self.coordinates.items():
            if not isinstance(node_id, int):
                raise ValueError("Düğüm kimlikleri tam sayı olmalıdır.")

            if len(point) != 2:
                raise ValueError(
                    f"Düğüm {node_id} iki koordinata sahip olmalıdır."
                )

            x, y = float(point[0]), float(point[1])

            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    f"Düğüm {node_id} koordinatları sonlu olmalıdır."
                )

            normalized[node_id] = (x, y)

        object.__setattr__(self, "coordinates", normalized)

    @property
    def node_ids(self) -> tuple[int, ...]:
        """Problemdeki düğüm kimliklerini sıralı olarak döndürür."""

        return tuple(sorted(self.coordinates))

    def to_dict(
        self,
        *,
        include_coordinates: bool = False,
    ) -> dict[str, Any]:
        """Problem bilgisini JSON'a yazılabilir sözlüğe dönüştürür."""

        result: dict[str, Any] = {
            "name": self.name,
            "source_type": self.source_type.value,
            "dimension": self.dimension,
            "depot_id": self.depot_id,
            "edge_weight_type": self.edge_weight_type,
            "node_ids": list(self.node_ids),
            "seed": self.seed,
            "source_file": (
                self.source_file.as_posix()
                if self.source_file is not None
                else None
            ),
            "optimal_tour_file": (
                self.optimal_tour_file.as_posix()
                if self.optimal_tour_file is not None
                else None
            ),
            "reference": (
                self.reference.to_dict()
                if self.reference is not None
                else None
            ),
        }

        if include_coordinates:
            result["coordinates"] = [
                {
                    "node_id": node_id,
                    "x": self.coordinates[node_id][0],
                    "y": self.coordinates[node_id][1],
                }
                for node_id in self.node_ids
            ]

        return result