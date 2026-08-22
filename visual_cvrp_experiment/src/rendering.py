"""CVRP problemleri için görsel üretim araçları."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .problem import CVRPProblem


class DemandEncoding(str, Enum):
    """Müşteri talebinin görselde gösterim yöntemi."""

    NUMERIC = "numeric"


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

    figure, axis = plt.subplots(
        figsize=(10.0, 7.5),
        dpi=dpi,
    )
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

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
        s=850,
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

        if normalized_encoding is DemandEncoding.NUMERIC:
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
        "Capacitated Vehicle Routing Problem",
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
        bottom=0.04,
        top=0.82,
    )
    figure.savefig(
        path,
        format="png",
        dpi=dpi,
        facecolor="white",
    )
    plt.close(figure)

    return path