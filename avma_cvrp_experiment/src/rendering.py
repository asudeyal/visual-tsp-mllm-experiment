"""Deterministic model-facing rendering for AVMA-CVRP.

Model-facing images may show only:
- customer locations and node IDs,
- depot marker,
- route connections,
- visual demand encodings,
- visual full-capacity reference markers,
- visual vehicle-availability markers.

They must not show numerical demands, capacity, route loads, distances, gaps,
optimums, validation status, or validation reasons.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Rectangle

from .config import (
    DemandEncodingConfig,
    RenderConfig,
    RouteRenderingConfig,
)
from .schemas import ProblemInstance


NODE_EDGE_COLOR = "#143A63"
NODE_FILL_COLOR = "#FFFFFF"
DEPOT_FILL_COLOR = "#111827"
DEMAND_FILL_COLOR = "#2563EB"


@dataclass(frozen=True)
class _RenderStyle:
    figure_size_inches: float
    base_node_size: float
    depot_size: float
    font_size: float
    node_line_width: float
    depot_line_width: float
    route_line_width: float


def _positive_min_distance(problem: ProblemInstance) -> float | None:
    """Return the smallest positive Euclidean coordinate distance."""

    best = math.inf
    node_ids = problem.node_ids

    for index, first in enumerate(node_ids):
        x1, y1 = problem.coordinates[first]

        for second in node_ids[index + 1:]:
            x2, y2 = problem.coordinates[second]
            distance = math.hypot(x2 - x1, y2 - y1)

            if 0.0 < distance < best:
                best = distance

    return None if math.isinf(best) else best


def _render_style(
    problem: ProblemInstance,
    cfg: RenderConfig,
) -> _RenderStyle:
    xs = [coordinate[0] for coordinate in problem.coordinates.values()]
    ys = [coordinate[1] for coordinate in problem.coordinates.values()]

    max_span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    node_count = max(problem.dimension, 1)

    # Larger instances receive a larger canvas instead of microscopic labels.
    figure_size_inches = max(
        cfg.figure_size_inches,
        min(
            12.0,
            cfg.figure_size_inches + max(0, node_count - 20) * 0.10,
        ),
    )

    # Matplotlib scatter sizes are in points². The closest-node cap prevents
    # obvious overlap while preserving the relative visual encoding.
    base_node_size = float(cfg.node_size)
    nearest_neighbor = _positive_min_distance(problem)

    if nearest_neighbor is not None:
        usable_axis_points = figure_size_inches * 72.0 * 0.78
        nearest_points = usable_axis_points * nearest_neighbor / max_span
        safe_diameter_points = nearest_points * 0.80
        base_node_size = min(base_node_size, safe_diameter_points**2)

    if node_count <= 40:
        font_size = float(max(9.5, cfg.font_size))
    elif node_count <= 70:
        font_size = float(max(9.5, cfg.font_size + 0.5))
    else:
        font_size = float(max(8.5, cfg.font_size - 0.5))

    node_line_width = 1.6 if node_count <= 40 else 1.45
    depot_line_width = node_line_width + 0.45
    route_line_width = max(
        1.1,
        cfg.route_line_width - (0.10 if node_count > 40 else 0.0),
    )

    return _RenderStyle(
        figure_size_inches=figure_size_inches,
        base_node_size=base_node_size,
        depot_size=base_node_size * 0.85,
        font_size=font_size,
        node_line_width=node_line_width,
        depot_line_width=depot_line_width,
        route_line_width=route_line_width,
    )


def _limits(
    problem: ProblemInstance,
    padding_ratio: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [coordinate[0] for coordinate in problem.coordinates.values()]
    ys = [coordinate[1] for coordinate in problem.coordinates.values()]

    x_span = max(max(xs) - min(xs), 1.0)
    y_span = max(max(ys) - min(ys), 1.0)

    x_padding = x_span * padding_ratio
    y_padding = y_span * padding_ratio

    return (
        (min(xs) - x_padding, max(xs) + x_padding),
        (min(ys) - y_padding, max(ys) + y_padding),
    )


def _needs_header(
    problem: ProblemInstance,
    demand_encoding: DemandEncodingConfig,
) -> bool:
    has_demand_legend = (
        demand_encoding.mode != "none"
        and demand_encoding.show_visual_legend
    )
    has_vehicle_icons = (
        demand_encoding.show_vehicle_icons
        and problem.max_vehicles is not None
    )
    return has_demand_legend or has_vehicle_icons


def _base_axes(
    problem: ProblemInstance,
    cfg: RenderConfig,
    demand_encoding: DemandEncodingConfig,
) -> tuple[plt.Figure, Axes, Axes | None, _RenderStyle]:
    style = _render_style(problem, cfg)
    has_header = _needs_header(problem, demand_encoding)

    figure_height = style.figure_size_inches + (0.85 if has_header else 0.0)
    fig = plt.figure(
        figsize=(style.figure_size_inches, figure_height),
        dpi=cfg.dpi,
    )
    fig.patch.set_facecolor("white")

    if has_header:
        grid = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=(0.85, style.figure_size_inches),
            hspace=0.03,
        )
        header_ax = fig.add_subplot(grid[0])
        ax = fig.add_subplot(grid[1])
    else:
        header_ax = None
        ax = fig.add_subplot(111)

    ax.set_facecolor("white")

    x_limits, y_limits = _limits(problem, cfg.padding_ratio)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if header_ax is not None:
        header_ax.set_facecolor("white")
        header_ax.set_xlim(0.0, 1.0)
        header_ax.set_ylim(0.0, 1.0)
        header_ax.axis("off")

    return fig, ax, header_ax, style


def _demand_ratio(problem: ProblemInstance, node: int) -> float:
    """Return hidden numerical demand ratio used only for visual encoding."""

    if node == problem.depot:
        return 0.0

    demand = problem.demands.get(node, 0)
    return min(1.0, max(0.0, demand / problem.capacity))


def _node_radius_data(
    style: _RenderStyle,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> float:
    """Approximate a scatter-marker radius in data coordinates."""

    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    usable_axis_points = style.figure_size_inches * 72.0 * 0.78

    data_per_point = min(x_span, y_span) / max(usable_axis_points, 1.0)
    scatter_radius_points = math.sqrt(style.base_node_size) * 0.50

    return max(data_per_point * scatter_radius_points, min(x_span, y_span) * 0.003)


def _label_color(fill_color: str) -> str:
    """Choose black or white text for a readable colored node."""

    red, green, blue = to_rgb(fill_color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.45 else "#111827"


def _demand_colormap(
    demand_encoding: DemandEncodingConfig,
) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "avma_cvrp_demand",
        list(demand_encoding.color_stops),
    )


def _draw_size_reference(
    ax: Axes,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> None:
    # Gerçek haritadaki (Axes) oranlarla lejant (Header Axes) oranlarını birebir eşitlemek
    # için marker alanları doğrudan style.base_node_size kullanılarak hesaplanmıştır.
    small_area = style.base_node_size * demand_encoding.size_min_factor
    full_area = style.base_node_size * demand_encoding.size_max_factor

    ax.scatter(
        [0.40, 0.56],
        [0.50, 0.50],
        s=[small_area, full_area],
        facecolors=DEMAND_FILL_COLOR,
        edgecolors=NODE_EDGE_COLOR,
        linewidths=1.3,
        transform=ax.transAxes,
        zorder=2,
    )


def _draw_bar_reference(
    ax: Axes,
    demand_encoding: DemandEncodingConfig,
) -> None:
    width_points = float(demand_encoding.bar_width_points)
    height_points = float(demand_encoding.bar_height_points)

    for center_x, full in ((0.40, False), (0.58, True)):
        drawing = DrawingArea(width_points, height_points, 0, 0, clip=False)

        outline = Rectangle(
            (0.0, 0.0),
            width_points,
            height_points,
            facecolor=NODE_FILL_COLOR,
            edgecolor=NODE_EDGE_COLOR,
            linewidth=1.0,
        )
        drawing.add_artist(outline)

        if full:
            fill = Rectangle(
                (0.0, 0.0),
                width_points,
                height_points,
                facecolor=DEMAND_FILL_COLOR,
                edgecolor="none",
            )
            drawing.add_artist(fill)

        artist = AnnotationBbox(
            drawing,
            (center_x, 0.50),
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0.0,
            zorder=3,
        )
        ax.add_artist(artist)

def _draw_dot_reference(
    ax: Axes,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> None:
    diameter_points = math.sqrt(style.base_node_size)
    padding_points = 1.5
    drawing_size = diameter_points + (2.0 * padding_points)
    marker_radius = diameter_points * 0.50

    for center_x, ratio in ((0.40, 0.0), (0.58, 1.0)):
        drawing = DrawingArea(drawing_size, drawing_size, 0, 0, clip=False)
        center = padding_points + marker_radius

        outline = Circle(
            (center, center),
            marker_radius,
            facecolor=NODE_FILL_COLOR,
            edgecolor=NODE_EDGE_COLOR,
            linewidth=style.node_line_width,
        )
        drawing.add_artist(outline)

        offsets = _dot_positions(
            node=0,
            ratio=ratio,
            radius=marker_radius,
            grid_size=demand_encoding.dot_grid_size,
        )

        for offset_x, offset_y in offsets:
            dot = Circle(
                (center + offset_x, center + offset_y),
                demand_encoding.dot_radius_points,
                facecolor=NODE_EDGE_COLOR,
                edgecolor="none",
            )
            drawing.add_artist(dot)

        artist = AnnotationBbox(
            drawing,
            (center_x, 0.50),
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0.0,
            zorder=3,
        )
        ax.add_artist(artist)

def _draw_color_reference(
    ax: Axes,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> None:
    # Only empty/full color endpoints.
    color_map = _demand_colormap(demand_encoding)
    ax.scatter(
        [0.40, 0.58],
        [0.50, 0.50],
        s=[style.base_node_size, style.base_node_size],
        facecolors=[color_map(0.0), color_map(1.0)],
        edgecolors=NODE_EDGE_COLOR,
        linewidths=1.2,
        transform=ax.transAxes,
        zorder=2,
    )

def _draw_visual_demand_legend(
    ax: Axes,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> None:
    """Draw a visual-only low-to-full reference with no numerical labels."""

    if demand_encoding.mode == "size":
        _draw_size_reference(ax, demand_encoding, style)
    elif demand_encoding.mode == "bar":
        _draw_bar_reference(ax, demand_encoding)
    elif demand_encoding.mode == "dot_density":
        _draw_dot_reference(ax, demand_encoding, style)
    elif demand_encoding.mode == "color":
        _draw_color_reference(ax, demand_encoding, style)


def _draw_vehicle_icons(
    ax: Axes,
    max_vehicles: int,
) -> None:
    """Draw one unlabeled visual vehicle marker per available vehicle."""

    display_count = min(max_vehicles, 16)
    total_width = min(0.34, 0.021 * display_count)
    icon_width = total_width / display_count
    start_x = 0.96 - total_width

    for index in range(display_count):
        x = start_x + index * icon_width

        body = Rectangle(
            (x, 0.43),
            icon_width * 0.72,
            0.16,
            transform=ax.transAxes,
            facecolor="#D1D5DB",
            edgecolor="#374151",
            linewidth=0.8,
            zorder=2,
        )
        cabin = Rectangle(
            (x + icon_width * 0.48, 0.54),
            icon_width * 0.22,
            0.08,
            transform=ax.transAxes,
            facecolor="#9CA3AF",
            edgecolor="#374151",
            linewidth=0.6,
            zorder=3,
        )

        ax.add_patch(body)
        ax.add_patch(cabin)

        ax.scatter(
            [x + icon_width * 0.15, x + icon_width * 0.57],
            [0.41, 0.41],
            s=10,
            color="#111827",
            transform=ax.transAxes,
            zorder=4,
        )


def _draw_header(
    ax: Axes | None,
    problem: ProblemInstance,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> None:
    if ax is None:
        return

    if (
        demand_encoding.mode != "none"
        and demand_encoding.show_visual_legend
    ):
        _draw_visual_demand_legend(ax, demand_encoding, style)

    if (
        demand_encoding.show_vehicle_icons
        and problem.max_vehicles is not None
    ):
        _draw_vehicle_icons(ax, problem.max_vehicles)


def _bar_dimensions_data(
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[float, float]:
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    usable_axis_points = style.figure_size_inches * 72.0 * 0.78

    width = x_span * demand_encoding.bar_width_points / usable_axis_points
    height = y_span * demand_encoding.bar_height_points / usable_axis_points

    return width, height


def _draw_bar_marker(
    ax: Axes,
    *,
    x: float,
    y: float,
    ratio: float,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    width, height = _bar_dimensions_data(
        style,
        demand_encoding,
        x_limits,
        y_limits,
    )

    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    usable_axis_points = style.figure_size_inches * 72.0 * 0.78
    data_per_point = min(x_span, y_span) / max(usable_axis_points, 1.0)
    bar_node_area = max(18.0, style.base_node_size * 0.08)
    node_radius = data_per_point * math.sqrt(bar_node_area) * 0.50

    left = x - width / 2.0
    bottom = max(
        y_limits[0],
        y - node_radius - (height * 1.25),
    )

    outline = Rectangle(
        (left, bottom),
        width,
        height,
        facecolor=NODE_FILL_COLOR,
        edgecolor=NODE_EDGE_COLOR,
        linewidth=1.0,
        zorder=4,
    )
    fill = Rectangle(
        (left, bottom),
        width * ratio,
        height,
        facecolor=DEMAND_FILL_COLOR,
        edgecolor="none",
        zorder=4.1,
    )

    ax.add_patch(outline)
    ax.add_patch(fill)

def _dot_positions(
    *,
    node: int,
    ratio: float,
    radius: float,
    grid_size: int,
) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    denominator = max(grid_size - 1, 1)

    for row in range(grid_size):
        for column in range(grid_size):
            relative_x = -0.78 + 1.56 * column / denominator
            relative_y = -0.78 + 1.56 * row / denominator
            distance = math.hypot(relative_x, relative_y)

            if distance <= 0.88:
                candidates.append(
                    (relative_x * radius, relative_y * radius)
                )

    generator = random.Random(f"avma-cvrp-dot-density-{node}")
    generator.shuffle(candidates)

    if ratio <= 0.0:
        count = 0
    else:
        count = max(1, int(round(ratio * len(candidates))))

    return candidates[:count]

def _draw_dot_density_marker(
    ax: Axes,
    *,
    node: int,
    x: float,
    y: float,
    ratio: float,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    radius = _node_radius_data(style, x_limits, y_limits)
    offsets = _dot_positions(
        node=node,
        ratio=ratio,
        radius=radius,
        grid_size=demand_encoding.dot_grid_size,
    )

    if not offsets:
        return

    xs = [x + offset_x for offset_x, _ in offsets]
    ys = [y + offset_y for _, offset_y in offsets]

    dot_area = max(4.0, (demand_encoding.dot_radius_points * 2.0) ** 2)

    ax.scatter(
        xs,
        ys,
        s=dot_area,
        facecolors=NODE_EDGE_COLOR,
        edgecolors="none",
        zorder=4,
    )


def _draw_customer_node(
    ax: Axes,
    problem: ProblemInstance,
    node: int,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    color_map: LinearSegmentedColormap | None,
) -> None:
    x, y = problem.coordinates[node]
    ratio = _demand_ratio(problem, node)

    marker_area = style.base_node_size
    fill_color = NODE_FILL_COLOR
    marker_edge_color = NODE_EDGE_COLOR
    marker_line_width = style.node_line_width

    if demand_encoding.mode == "size":
        scale = (
            demand_encoding.size_min_factor
            + (
                demand_encoding.size_max_factor
                - demand_encoding.size_min_factor
            )
            * ratio
        )
        marker_area = style.base_node_size * scale
        fill_color = DEMAND_FILL_COLOR

    elif demand_encoding.mode == "color" and color_map is not None:
        fill_color = color_map(ratio)

    elif demand_encoding.mode == "bar":
        # Position only. Demand is encoded exclusively by the bar.
        marker_area = max(18.0, style.base_node_size * 0.08)
        fill_color = NODE_EDGE_COLOR
        marker_edge_color = NODE_EDGE_COLOR
        marker_line_width = 0.8

    ax.scatter(
        [x],
        [y],
        s=marker_area,
        marker="o",
        facecolors=fill_color,
        edgecolors=marker_edge_color,
        linewidths=marker_line_width,
        zorder=3,
    )

    if demand_encoding.mode == "bar":
        _draw_bar_marker(
            ax,
            x=x,
            y=y,
            ratio=ratio,
            style=style,
            demand_encoding=demand_encoding,
            x_limits=x_limits,
            y_limits=y_limits,
        )

    elif demand_encoding.mode == "dot_density":
        _draw_dot_density_marker(
            ax,
            node=node,
            x=x,
            y=y,
            ratio=ratio,
            style=style,
            demand_encoding=demand_encoding,
            x_limits=x_limits,
            y_limits=y_limits,
        )

    # Same ID presentation for every encoding.
    label = ax.annotate(
        str(node),
        xy=(x, y),
        xytext=(4.0, 4.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=style.font_size,
        color="#111827",
        weight="bold",
        zorder=6,
    )
    label.set_path_effects(
        [path_effects.withStroke(linewidth=2.0, foreground="white")]
    )

def _draw_depot(
    ax: Axes,
    problem: ProblemInstance,
    style: _RenderStyle,
) -> None:
    x, y = problem.coordinates[problem.depot]

    ax.scatter(
        [x],
        [y],
        s=style.depot_size,
        marker="s",
        facecolors=DEPOT_FILL_COLOR,
        edgecolors="#000000",
        linewidths=style.depot_line_width,
        zorder=4,
    )

    ax.text(
        x,
        y,
        str(problem.depot),
        ha="center",
        va="center",
        fontsize=style.font_size,
        color="white",
        weight="bold",
        zorder=5,
    )


def _draw_nodes(
    ax: Axes,
    problem: ProblemInstance,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
) -> None:
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    color_map = (
        _demand_colormap(demand_encoding)
        if demand_encoding.mode == "color"
        else None
    )

    for node in problem.node_ids:
        if node == problem.depot:
            continue

        _draw_customer_node(
            ax,
            problem,
            node,
            style,
            demand_encoding,
            x_limits,
            y_limits,
            color_map,
        )

    _draw_depot(ax, problem, style)


def _draw_routes(
    ax: Axes,
    problem: ProblemInstance,
    routes: Sequence[Iterable[int]],
    style: _RenderStyle,
    route_rendering: RouteRenderingConfig,
) -> None:
    routes_tuple = tuple(tuple(route) for route in routes)
    palette = route_rendering.palette

    for route_index, route in enumerate(routes_tuple):
        color = palette[route_index % len(palette)]

        for first, second in zip(route, route[1:]):
            ax.plot(
                [
                    problem.coordinates[first][0],
                    problem.coordinates[second][0],
                ],
                [
                    problem.coordinates[first][1],
                    problem.coordinates[second][1],
                ],
                color=color,
                linewidth=style.route_line_width,
                zorder=1,
            )


def _validate_renderable_nodes(
    problem: ProblemInstance,
    routes: Sequence[Iterable[int]],
) -> None:
    unknown_nodes = sorted(
        {
            node
            for route in routes
            for node in route
            if node not in problem.coordinates
        }
    )

    if unknown_nodes:
        raise ValueError(
            "Render edilemeyen bilinmeyen node ID'leri: "
            f"{unknown_nodes}"
        )


def render_problem(
    problem: ProblemInstance,
    output_path: str | Path,
    cfg: RenderConfig,
    *,
    demand_encoding: DemandEncodingConfig | None = None,
) -> Path:
    """Render the shared model-facing original CVRP problem image."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active_encoding = demand_encoding or DemandEncodingConfig()

    fig, ax, header_ax, style = _base_axes(
        problem,
        cfg,
        active_encoding,
    )
    _draw_header(header_ax, problem, active_encoding, style)
    _draw_nodes(ax, problem, style, active_encoding)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return output_path


def render_routes(
    problem: ProblemInstance,
    routes: Sequence[Iterable[int]],
    output_path: str | Path,
    cfg: RenderConfig,
    *,
    demand_encoding: DemandEncodingConfig | None = None,
    route_rendering: RouteRenderingConfig | None = None,
) -> Path:
    """Render model-facing CVRP route image without numerical diagnostics."""

    routes_tuple = tuple(tuple(route) for route in routes)
    _validate_renderable_nodes(problem, routes_tuple)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active_encoding = demand_encoding or DemandEncodingConfig()
    active_route_rendering = route_rendering or RouteRenderingConfig()

    fig, ax, header_ax, style = _base_axes(
        problem,
        cfg,
        active_encoding,
    )
    _draw_header(header_ax, problem, active_encoding, style)
    _draw_nodes(ax, problem, style, active_encoding)
    _draw_routes(
        ax,
        problem,
        routes_tuple,
        style,
        active_route_rendering,
    )
    _draw_nodes(ax, problem, style, active_encoding)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return output_path


def render_diagnostic_routes(
    problem: ProblemInstance,
    routes: Sequence[Iterable[int]],
    output_path: str | Path,
    cfg: RenderConfig,
    *,
    demand_encoding: DemandEncodingConfig | None = None,
    route_rendering: RouteRenderingConfig | None = None,
) -> Path:
    """Render researcher-only numerical diagnostics.

    Never pass this image to any provider or agent prompt.
    """

    from .evaluation import validate_cvrp_routes

    routes_tuple = tuple(tuple(route) for route in routes)
    _validate_renderable_nodes(problem, routes_tuple)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active_encoding = demand_encoding or DemandEncodingConfig()
    active_route_rendering = route_rendering or RouteRenderingConfig()

    fig, ax, header_ax, style = _base_axes(
        problem,
        cfg,
        active_encoding,
    )
    _draw_header(header_ax, problem, active_encoding, style)
    _draw_nodes(ax, problem, style, active_encoding)
    _draw_routes(
        ax,
        problem,
        routes_tuple,
        style,
        active_route_rendering,
    )
    _draw_nodes(ax, problem, style, active_encoding)

    validation = validate_cvrp_routes(problem, routes_tuple)

    route_load_text = ", ".join(
        f"R{index + 1}: {load}/{problem.capacity}"
        for index, load in enumerate(validation.route_loads)
    )
    vehicle_limit = (
        "unlimited"
        if problem.max_vehicles is None
        else str(problem.max_vehicles)
    )
    status = "valid" if validation.valid else ", ".join(validation.reasons)

    fig.text(
        0.5,
        0.015,
        (
            f"Route loads: {route_load_text}   |   "
            f"Vehicles: {validation.vehicle_count}/{vehicle_limit}   |   "
            f"Validation: {status}"
        ),
        ha="center",
        va="bottom",
        fontsize=max(7.0, style.font_size - 1.0),
        color="#111827",
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)

    return output_path