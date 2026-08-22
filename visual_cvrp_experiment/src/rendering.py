"""CVRP problemleri ve çözümleri için görsel üretim araçları."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .problem import CVRPProblem


class DemandEncoding(str, Enum):
    """Müşteri talebinin görselde gösterim yöntemi."""

    NUMERIC = "numeric"
    SIZE = "size"


_ROUTE_COLORS = (
    "#E63946",
    "#2A9D8F",
    "#F4A261",
    "#7B2CBF",
    "#3A86FF",
    "#8A5A44",
)

_NUMERIC_MARKER_AREA = 850.0
_SIZE_MARKER_AREA_PER_DEMAND_UNIT = 500.0


def _normalize_encoding(
    encoding: DemandEncoding | str,
) -> DemandEncoding:
    try:
        return DemandEncoding(encoding)
    except ValueError as error:
        supported = ", ".join(
            item.value
            for item in DemandEncoding
        )
        raise ValueError(
            "Desteklenmeyen talep gösterimi: "
            f"{encoding!r}. Desteklenenler: {supported}"
        ) from error


def _validate_output(
    output_path: Path | str,
    *,
    dpi: int,
) -> Path:
    if dpi < 1:
        raise ValueError(
            "DPI değeri pozitif olmalıdır."
        )

    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError(
            "Görsel çıktı uzantısı .png olmalıdır."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def _plot_bounds(
    problem: CVRPProblem,
) -> tuple[float, float, float, float]:
    x_values = [
        node.x
        for node in problem.nodes
    ]
    y_values = [
        node.y
        for node in problem.nodes
    ]

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    largest_span = max(
        x_max - x_min,
        y_max - y_min,
        1.0,
    )
    padding = max(
        8.0,
        largest_span * 0.12,
    )

    return (
        x_min - padding,
        x_max + padding,
        y_min - padding,
        y_max + padding,
    )


def _create_canvas(
    *,
    dpi: int,
):
    figure, axis = plt.subplots(
        figsize=(10.0, 7.5),
        dpi=dpi,
    )
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    return figure, axis


def _customer_marker_areas(
    problem: CVRPProblem,
    *,
    encoding: DemandEncoding,
) -> list[float]:
    if encoding is DemandEncoding.SIZE:
        return [
            (
                _SIZE_MARKER_AREA_PER_DEMAND_UNIT
                * customer.demand
            )
            for customer in problem.customers
        ]

    return [
        _NUMERIC_MARKER_AREA
        for _ in problem.customers
    ]


def _draw_nodes(
    axis,
    problem: CVRPProblem,
    *,
    encoding: DemandEncoding,
) -> None:
    customers = problem.customers

    axis.scatter(
        [
            customer.x
            for customer in customers
        ],
        [
            customer.y
            for customer in customers
        ],
        s=_customer_marker_areas(
            problem,
            encoding=encoding,
        ),
        marker="o",
        color="#2F80ED",
        edgecolors="#12355B",
        linewidths=1.8,
        zorder=3,
    )

    for customer in customers:
        axis.text(
            customer.x,
            customer.y,
            str(customer.node_id),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
            fontweight="bold",
            zorder=4,
        )

        if encoding is DemandEncoding.NUMERIC:
            axis.annotate(
                f"d={customer.demand}",
                xy=(customer.x, customer.y),
                xytext=(0, 19),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=11,
                fontweight="bold",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#6B7280",
                    "linewidth": 1.0,
                },
                zorder=5,
            )

    depot = problem.depot

    axis.scatter(
        [depot.x],
        [depot.y],
        s=1050,
        marker="s",
        color="#111827",
        edgecolors="black",
        linewidths=2.0,
        zorder=3,
    )
    axis.text(
        depot.x,
        depot.y,
        str(depot.node_id),
        horizontalalignment="center",
        verticalalignment="center",
        color="white",
        fontsize=13,
        fontweight="bold",
        zorder=4,
    )
    axis.annotate(
        "DEPOT",
        xy=(depot.x, depot.y),
        xytext=(0, -23),
        textcoords="offset points",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        zorder=5,
    )


def _draw_header(
    figure,
    problem: CVRPProblem,
    *,
    title: str,
) -> None:
    vehicle_count_text = (
        str(problem.vehicle_count)
        if problem.vehicle_count is not None
        else "not fixed"
    )
    information = (
        f"Vehicle capacity: Q = "
        f"{problem.vehicle_capacity}"
        "    |    "
        f"Available vehicles: K = "
        f"{vehicle_count_text}"
    )

    figure.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.965,
        color="#111827",
    )
    figure.text(
        0.5,
        0.895,
        information,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F3F4F6",
            "edgecolor": "#4B5563",
            "linewidth": 1.2,
        },
        zorder=6,
    )


def _save_figure(
    figure,
    axis,
    problem: CVRPProblem,
    path: Path,
    *,
    dpi: int,
    bottom: float,
) -> None:
    x_min, x_max, y_min, y_max = _plot_bounds(
        problem
    )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect(
        "equal",
        adjustable="box",
    )
    axis.axis("off")

    figure.subplots_adjust(
        left=0.03,
        right=0.97,
        bottom=bottom,
        top=0.82,
    )
    figure.savefig(
        path,
        format="png",
        dpi=dpi,
        facecolor="white",
    )
    plt.close(figure)


def render_problem(
    problem: CVRPProblem,
    output_path: Path | str,
    *,
    encoding: DemandEncoding | str = (
        DemandEncoding.NUMERIC
    ),
    dpi: int = 160,
) -> Path:
    """CVRP problem görselini PNG olarak oluştur."""

    normalized_encoding = _normalize_encoding(
        encoding
    )
    path = _validate_output(
        output_path,
        dpi=dpi,
    )
    figure, axis = _create_canvas(dpi=dpi)

    _draw_nodes(
        axis,
        problem,
        encoding=normalized_encoding,
    )
    _draw_header(
        figure,
        problem,
        title="Capacitated Vehicle Routing Problem",
    )
    _save_figure(
        figure,
        axis,
        problem,
        path,
        dpi=dpi,
        bottom=0.04,
    )
    return path


def render_solution(
    problem: CVRPProblem,
    routes: Sequence[Sequence[int]],
    output_path: Path | str,
    *,
    title: str,
    route_loads: Sequence[int] | None = None,
    encoding: DemandEncoding | str = (
        DemandEncoding.NUMERIC
    ),
    dpi: int = 160,
) -> Path:
    """Bir CVRP çözümünü farklı araç renkleriyle çiz."""

    if not title.strip():
        raise ValueError(
            "Çözüm görseli başlığı boş olamaz."
        )

    normalized_encoding = _normalize_encoding(
        encoding
    )
    path = _validate_output(
        output_path,
        dpi=dpi,
    )
    normalized_routes = tuple(
        tuple(route)
        for route in routes
    )

    if not normalized_routes:
        raise ValueError(
            "En az bir rota bulunmalıdır."
        )

    known_node_ids = {
        node.node_id
        for node in problem.nodes
    }

    for route_index, route in enumerate(
        normalized_routes,
        start=1,
    ):
        if len(route) < 2:
            raise ValueError(
                f"Rota {route_index} en az iki düğüm "
                "içermelidir."
            )

        unknown_node_ids = sorted(
            {
                node_id
                for node_id in route
                if node_id not in known_node_ids
            }
        )
        if unknown_node_ids:
            raise ValueError(
                f"Rota {route_index} bilinmeyen düğümler "
                f"içeriyor: {unknown_node_ids}"
            )

    if route_loads is None:
        normalized_loads = tuple(
            sum(
                problem.node(node_id).demand
                for node_id in route
                if node_id != problem.depot.node_id
            )
            for route in normalized_routes
        )
    else:
        normalized_loads = tuple(route_loads)
        if len(normalized_loads) != len(
            normalized_routes
        ):
            raise ValueError(
                "Rota yüklerinin sayısı rota sayısıyla "
                "eşleşmelidir."
            )

    figure, axis = _create_canvas(dpi=dpi)
    legend_handles = []

    for route_index, (route, load) in enumerate(
        zip(
            normalized_routes,
            normalized_loads,
            strict=True,
        ),
        start=1,
    ):
        color = _ROUTE_COLORS[
            (route_index - 1) % len(_ROUTE_COLORS)
        ]
        axis.plot(
            [
                problem.node(node_id).x
                for node_id in route
            ],
            [
                problem.node(node_id).y
                for node_id in route
            ],
            color=color,
            linewidth=2.8,
            alpha=0.9,
            solid_capstyle="round",
            zorder=1,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=3.0,
                label=(
                    f"Vehicle {route_index} — "
                    f"load {load}/"
                    f"{problem.vehicle_capacity}"
                ),
            )
        )

    _draw_nodes(
        axis,
        problem,
        encoding=normalized_encoding,
    )
    _draw_header(
        figure,
        problem,
        title=title,
    )
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, len(legend_handles)),
        frameon=True,
        facecolor="white",
        edgecolor="#9CA3AF",
        fontsize=9,
    )
    _save_figure(
        figure,
        axis,
        problem,
        path,
        dpi=dpi,
        bottom=0.14,
    )
    return path
