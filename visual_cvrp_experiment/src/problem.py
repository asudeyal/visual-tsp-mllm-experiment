"""CVRP problem tanımı ve temel alan doğrulamaları."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, isfinite
from typing import Any


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class Node:
    """Depo veya müşteri düğümü."""

    node_id: int
    x: float
    y: float
    demand: int = 0

    def __post_init__(self) -> None:
        if not _is_integer(self.node_id) or self.node_id < 0:
            raise ValueError(
                "node_id sıfır veya pozitif bir tam sayı olmalıdır."
            )
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError(
                "Düğüm koordinatları sonlu olmalıdır."
            )
        if not _is_integer(self.demand) or self.demand < 0:
            raise ValueError(
                "Talep sıfır veya pozitif bir tam sayı olmalıdır."
            )


@dataclass(frozen=True, slots=True)
class CVRPProblem:
    """Tek depolu ve özdeş araçlı CVRP örneği."""

    name: str
    depot: Node
    customers: tuple[Node, ...]
    vehicle_capacity: int
    vehicle_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Problem adı boş olamaz.")

        if not self.customers:
            raise ValueError("En az bir müşteri bulunmalıdır.")

        if (
            not _is_integer(self.vehicle_capacity)
            or self.vehicle_capacity < 1
        ):
            raise ValueError(
                "Araç kapasitesi pozitif bir tam sayı olmalıdır."
            )

        if self.depot.demand != 0:
            raise ValueError("Depo talebi sıfır olmalıdır.")

        node_ids = [
            node.node_id
            for node in self.nodes
        ]

        if len(node_ids) != len(set(node_ids)):
            raise ValueError(
                "Düğüm kimlikleri benzersiz olmalıdır."
            )

        for customer in self.customers:
            if customer.demand < 1:
                raise ValueError(
                    "Müşteri talebi pozitif olmalıdır."
                )

            if customer.demand > self.vehicle_capacity:
                raise ValueError(
                    "Tek bir müşterinin talebi "
                    "araç kapasitesini aşamaz."
                )

        if self.vehicle_count is not None:
            if (
                not _is_integer(self.vehicle_count)
                or self.vehicle_count < 1
            ):
                raise ValueError(
                    "Araç sayısı pozitif bir tam sayı olmalıdır."
                )

            if (
                self.vehicle_count
                < self.vehicle_count_lower_bound
            ):
                raise ValueError(
                    "Araç sayısı toplam talebi taşımak "
                    "için kesinlikle yetersizdir."
                )

    @property
    def nodes(self) -> tuple[Node, ...]:
        return (self.depot, *self.customers)

    @property
    def dimension(self) -> int:
        return len(self.nodes)

    @property
    def customer_count(self) -> int:
        return len(self.customers)

    @property
    def customer_ids(self) -> tuple[int, ...]:
        return tuple(
            customer.node_id
            for customer in self.customers
        )

    @property
    def total_demand(self) -> int:
        return sum(
            customer.demand
            for customer in self.customers
        )

    @property
    def vehicle_count_lower_bound(self) -> int:
        """Yalnızca toplam talebe dayalı araç alt sınırı."""

        return ceil(
            self.total_demand
            / self.vehicle_capacity
        )

    def node(self, node_id: int) -> Node:
        for node in self.nodes:
            if node.node_id == node_id:
                return node

        raise KeyError(
            f"Bilinmeyen düğüm kimliği: {node_id}"
        )

    def distance(
        self,
        first_id: int,
        second_id: int,
    ) -> float:
        first = self.node(first_id)
        second = self.node(second_id)

        return hypot(
            first.x - second.x,
            first.y - second.y,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "customer_count": self.customer_count,
            "depot_id": self.depot.node_id,
            "vehicle_capacity": self.vehicle_capacity,
            "vehicle_count": self.vehicle_count,
            "total_demand": self.total_demand,
            "vehicle_count_lower_bound": (
                self.vehicle_count_lower_bound
            ),
            "nodes": [
                {
                    "id": node.node_id,
                    "x": node.x,
                    "y": node.y,
                    "demand": node.demand,
                }
                for node in self.nodes
            ],
        }