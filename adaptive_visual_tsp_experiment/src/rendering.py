"""Deterministic model-facing rendering.

No objective values, iteration numbers, gaps, validation labels or hidden feedback are
rendered on model-facing images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RenderConfig
from .schemas import ProblemInstance


@dataclass(frozen=True)
class _RenderStyle:
    figure_size_inches: float
    node_size: float
    depot_size: float
    font_size: float
    node_line_width: float
    depot_line_width: float
    route_line_width: float


def _positive_min_distance(problem: ProblemInstance) -> float | None:
    best = math.inf
    ids = problem.node_ids
    for index, first in enumerate(ids):
        x1, y1 = problem.coordinates[first]
        for second in ids[index + 1:]:
            x2, y2 = problem.coordinates[second]
            distance = math.hypot(x2 - x1, y2 - y1)
            if 0.0 < distance < best:
                best = distance
    return None if math.isinf(best) else best


def _render_style(problem: ProblemInstance, cfg: RenderConfig) -> _RenderStyle:
    xs = [xy[0] for xy in problem.coordinates.values()]
    ys = [xy[1] for xy in problem.coordinates.values()]
    max_span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    node_count = max(problem.dimension, 1)

    # Larger instances get a larger canvas rather than microscopic labels.
    figure_size_inches = max(
        cfg.figure_size_inches,
        min(12.0, cfg.figure_size_inches + max(0, node_count - 20) * 0.10),
    )

    # scatter sizes are expressed in points^2, so marker size stays visually
    # stable instead of changing with TSP coordinate units.  The closest pair
    # only caps the marker size enough to avoid obvious overlap.
    node_size = float(cfg.node_size)
    nearest_neighbor = _positive_min_distance(problem)
    if nearest_neighbor is not None:
        usable_axis_points = figure_size_inches * 72.0 * 0.88
        nearest_points = usable_axis_points * nearest_neighbor / max_span
        safe_diameter_points = nearest_points * 0.80
        node_size = min(node_size, safe_diameter_points ** 2)

    # Keep labels legible for 50--100 node experiments.  Resolution/canvas
    # growth handles density; font size should not collapse to 6 px-like text.
    if node_count <= 40:
        font_size = float(max(9.5, cfg.font_size))
    elif node_count <= 70:
        font_size = float(max(9.5, cfg.font_size + 0.5))
    else:
        font_size = float(max(8.5, cfg.font_size - 0.5))

    node_line_width = 1.6 if node_count <= 40 else 1.45
    depot_line_width = node_line_width + 0.45
    route_line_width = max(1.1, cfg.route_line_width - (0.10 if node_count > 40 else 0.0))

    return _RenderStyle(
        figure_size_inches=figure_size_inches,
        node_size=node_size,
        depot_size=node_size * 1.12,
        font_size=font_size,
        node_line_width=node_line_width,
        depot_line_width=depot_line_width,
        route_line_width=route_line_width,
    )


def _limits(
    problem: ProblemInstance,
    padding_ratio: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [xy[0] for xy in problem.coordinates.values()]
    ys = [xy[1] for xy in problem.coordinates.values()]
    x_span = max(max(xs) - min(xs), 1.0)
    y_span = max(max(ys) - min(ys), 1.0)
    x_pad = x_span * padding_ratio
    y_pad = y_span * padding_ratio
    return (
        (min(xs) - x_pad, max(xs) + x_pad),
        (min(ys) - y_pad, max(ys) + y_pad),
    )


def _draw_nodes(
    ax: plt.Axes,
    problem: ProblemInstance,
    style: _RenderStyle,
) -> None:
    normal = [node for node in problem.node_ids if node != problem.depot]
    if normal:
        xs = [problem.coordinates[node][0] for node in normal]
        ys = [problem.coordinates[node][1] for node in normal]
        ax.scatter(
            xs,
            ys,
            s=style.node_size,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=style.node_line_width,
            zorder=3,
        )

    dx, dy = problem.coordinates[problem.depot]
    ax.scatter(
        [dx],
        [dy],
        s=style.depot_size,
        marker="s",
        facecolors="white",
        edgecolors="black",
        linewidths=style.depot_line_width,
        zorder=4,
    )

    for node in problem.node_ids:
        x, y = problem.coordinates[node]
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=style.font_size,
            color="black",
            zorder=5,
        )


def _base_axes(
    problem: ProblemInstance,
    cfg: RenderConfig,
) -> tuple[plt.Figure, plt.Axes, _RenderStyle]:
    style = _render_style(problem, cfg)
    fig, ax = plt.subplots(
        figsize=(style.figure_size_inches, style.figure_size_inches),
        dpi=cfg.dpi,
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    xlim, ylim = _limits(problem, cfg.padding_ratio)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax, style


def render_problem(
    problem: ProblemInstance,
    output_path: str | Path,
    cfg: RenderConfig,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, style = _base_axes(problem, cfg)
    _draw_nodes(ax, problem, style)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def render_route(
    problem: ProblemInstance,
    route: Iterable[int],
    output_path: str | Path,
    cfg: RenderConfig,
) -> Path:
    route = tuple(route)
    unknown = [node for node in route if node not in problem.coordinates]
    if unknown:
        raise ValueError(f"Render edilemeyen bilinmeyen node ID'leri: {sorted(set(unknown))}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, style = _base_axes(problem, cfg)

    for first, second in zip(route, route[1:]):
        ax.plot(
            [problem.coordinates[first][0], problem.coordinates[second][0]],
            [problem.coordinates[first][1], problem.coordinates[second][1]],
            color="black",
            linewidth=style.route_line_width,
            zorder=1,
        )

    _draw_nodes(ax, problem, style)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path
