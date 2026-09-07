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

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
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
from PIL import Image

from .config import DemandEncodingConfig, RenderConfig, RouteRenderingConfig
from .schemas import ProblemInstance

NODE_EDGE_COLOR = "#143A63"
NODE_FILL_COLOR = "#FFFFFF"
DEPOT_FILL_COLOR = "#111827"
DEMAND_FILL_COLOR = "#2563EB"

@dataclass(frozen=True)
class _RenderStyle:
    figure_size_inches: float
    primitive_scale: float
    bar_scale: float
    label_clearance_scale: float
    vehicle_icon_width_points: float
    vehicle_icon_height_points: float
    vehicle_text_size: float
    base_node_size: float
    depot_size: float
    font_size: float
    node_line_width: float
    depot_line_width: float
    route_line_width: float

Rect = tuple[float, float, float, float]

def _positive_min_distance(problem: ProblemInstance) -> float | None:
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


def _customer_count(problem: ProblemInstance) -> int:
    return sum(1 for node in problem.node_ids if node != problem.depot)


def _median_nearest_neighbor_ratio(problem: ProblemInstance) -> float:
    customers = [node for node in problem.node_ids if node != problem.depot]
    if len(customers) <= 1:
        return 1.0

    xs = [problem.coordinates[node][0] for node in customers]
    ys = [problem.coordinates[node][1] for node in customers]
    x_span = max(max(xs) - min(xs), 1.0)
    y_span = max(max(ys) - min(ys), 1.0)
    diagonal = math.hypot(x_span, y_span)
    if diagonal <= 0.0:
        return 1.0

    nearest_distances: list[float] = []
    for node in customers:
        x, y = problem.coordinates[node]
        best = math.inf
        for other in customers:
            if other == node:
                continue
            ox, oy = problem.coordinates[other]
            best = min(best, math.hypot(ox - x, oy - y))
        if math.isfinite(best) and best > 0.0:
            nearest_distances.append(best)

    if not nearest_distances:
        return 1.0

    nearest_distances.sort()
    median = nearest_distances[len(nearest_distances) // 2]
    return median / diagonal


def _density_penalty(problem: ProblemInstance) -> float:
    ratio = _median_nearest_neighbor_ratio(problem)
    if ratio < 0.018:
        return 0.86
    if ratio < 0.028:
        return 0.90
    if ratio < 0.040:
        return 0.94
    if ratio < 0.055:
        return 0.97
    return 1.00


def _primitive_scale(problem: ProblemInstance) -> float:
    """Deterministic dense-render scaling from node count and spatial density."""
    customer_count = _customer_count(problem)

    if customer_count <= 60:
        base_scale = 1.00
    elif customer_count <= 100:
        base_scale = 0.90
    elif customer_count <= 150:
        base_scale = 0.84
    elif customer_count <= 250:
        base_scale = 0.78
    elif customer_count <= 400:
        base_scale = 0.72
    else:
        base_scale = 0.68

    scale = base_scale * _density_penalty(problem)
    return max(0.62, min(1.00, scale))

def _render_style(problem: ProblemInstance, cfg: RenderConfig) -> _RenderStyle:
    # Fixed visual primitives for the frozen-pixel protocol.
    if cfg.fixed_canvas:
        header_height_px = min(
            max(int(cfg.panel_header_height_px), 0),
            max(int(cfg.canvas_height_px) - 1, 0),
        )
        map_viewport_px = min(
            int(cfg.map_width_px),
            int(cfg.canvas_height_px) - header_height_px,
        )
        map_size_inches = map_viewport_px / cfg.dpi
    else:
        map_size_inches = cfg.figure_size_inches

    primitive_scale = _primitive_scale(problem)
    customer_count = _customer_count(problem)

    # Bars should shrink a little more aggressively than labels/nodes on dense maps.
    if customer_count <= 80:
        bar_scale = primitive_scale
    else:
        bar_scale = max(0.56, primitive_scale * 0.92)

    label_clearance_scale = 1.0 + max(0.0, 1.0 - primitive_scale) * 0.80

    # Matplotlib scatter sizes are areas in pt^2, so square the linear
    # scale to shrink marker diameters by the same factor as bars/fonts.
    base_node_area = min(float(cfg.node_size), 360.0)
    base_node_size = base_node_area * primitive_scale * primitive_scale

    base_font_size = float(max(9.5, cfg.font_size))
    font_size = max(6.8, base_font_size * max(0.72, primitive_scale))

    vehicle_icon_width_points = max(18.0, 22.0 * max(0.92, primitive_scale))
    vehicle_icon_height_points = max(11.0, 13.0 * max(0.92, primitive_scale))
    vehicle_text_size = max(10.0, 12.0 * max(0.92, primitive_scale))

    node_line_width = 1.45
    depot_line_width = 1.90
    route_line_width = float(cfg.route_line_width)

    return _RenderStyle(
        figure_size_inches=map_size_inches,
        primitive_scale=primitive_scale,
        bar_scale=bar_scale,
        label_clearance_scale=label_clearance_scale,
        vehicle_icon_width_points=vehicle_icon_width_points,
        vehicle_icon_height_points=vehicle_icon_height_points,
        vehicle_text_size=vehicle_text_size,
        base_node_size=base_node_size,
        depot_size=max(
            base_node_size * 0.85,
            180.0,
        ),
        font_size=font_size,
        node_line_width=node_line_width,
        depot_line_width=depot_line_width,
        route_line_width=route_line_width,
    )

def _limits(
    problem: ProblemInstance,
    padding_ratio: float,
    *,
    square: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [coordinate[0] for coordinate in problem.coordinates.values()]
    ys = [coordinate[1] for coordinate in problem.coordinates.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)

    if not square:
        x_padding = x_span * padding_ratio
        y_padding = y_span * padding_ratio
        return (
            (x_min - x_padding, x_max + x_padding),
            (y_min - y_padding, y_max + y_padding),
        )

    # Fill the 1280x1280 viewport without distorting original coordinates.
    padded_x_span = x_span * (1.0 + 2.0 * padding_ratio)
    padded_y_span = y_span * (1.0 + 2.0 * padding_ratio)
    square_span = max(padded_x_span, padded_y_span)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    half = square_span / 2.0
    return (
        (center_x - half, center_x + half),
        (center_y - half, center_y + half),
    )

def _needs_header(problem: ProblemInstance, demand_encoding: DemandEncodingConfig) -> bool:
    return (
        demand_encoding.mode != "none" and demand_encoding.show_visual_legend
    ) or (
        demand_encoding.show_vehicle_icons and problem.max_vehicles is not None
    )

def _demand_placement(demand_encoding: DemandEncodingConfig) -> str:
    """Resolve generic demand placement while preserving legacy configs."""
    placement = getattr(demand_encoding, "placement", None)
    if placement is not None:
        return placement
    if (
        demand_encoding.mode == "bar"
        and demand_encoding.bar_layout == "side_panel"
    ):
        return "side_panel"
    return "collision_aware"


def _uses_side_panel(demand_encoding: DemandEncodingConfig) -> bool:
    return (
        demand_encoding.mode != "none"
        and _demand_placement(demand_encoding) == "side_panel"
    )


def _base_axes(
    problem: ProblemInstance,
    cfg: RenderConfig,
    demand_encoding: DemandEncodingConfig,
) -> tuple[plt.Figure, Axes, Axes | None, Axes | None, _RenderStyle]:
    style = _render_style(problem, cfg)

    if cfg.fixed_canvas:
        header_height_px = min(
            max(int(cfg.panel_header_height_px), 0),
            max(int(cfg.canvas_height_px) - 1, 0),
        )
        map_block_width_px = int(cfg.map_width_px)
        map_viewport_px = min(
            map_block_width_px,
            int(cfg.canvas_height_px) - header_height_px,
        )
        map_left_px = (map_block_width_px - map_viewport_px) / 2.0

        uses_side_panel = _uses_side_panel(demand_encoding)
        canvas_width_px = (
            int(cfg.canvas_width_px)
            if uses_side_panel
            else map_block_width_px
        )

        fig = plt.figure(
            figsize=(
                canvas_width_px / cfg.dpi,
                cfg.canvas_height_px / cfg.dpi,
            ),
            dpi=cfg.dpi,
        )
        fig.patch.set_facecolor("white")

        # Dedicated header above the map block. The map never occupies it.
        header_ax = fig.add_axes(
            [
                0.0,
                (cfg.canvas_height_px - header_height_px)
                / cfg.canvas_height_px,
                map_block_width_px / canvas_width_px,
                header_height_px / cfg.canvas_height_px,
            ],
            zorder=20,
        )
        header_ax.set_facecolor("white")
        header_ax.set_xlim(0.0, 1.0)
        header_ax.set_ylim(0.0, 1.0)
        header_ax.axis("off")

        # Same square map viewport for local, collision-aware and side-panel.
        ax = fig.add_axes(
            [
                map_left_px / canvas_width_px,
                0.0,
                map_viewport_px / canvas_width_px,
                map_viewport_px / cfg.canvas_height_px,
            ]
        )

        side_ax = None
        if uses_side_panel:
            # The side panel is demand-only and uses the full canvas height.
            side_ax = fig.add_axes(
                [
                    map_block_width_px / canvas_width_px,
                    0.0,
                    cfg.panel_width_px / canvas_width_px,
                    1.0,
                ]
            )
            side_ax.set_facecolor("white")
            side_ax.set_xlim(0.0, 1.0)
            side_ax.set_ylim(0.0, 1.0)
            side_ax.axis("off")
            side_ax.plot(
                [0.0, 0.0],
                [0.0, 1.0],
                color="#D1D5DB",
                linewidth=0.8,
                transform=side_ax.transAxes,
                clip_on=False,
            )

        x_limits, y_limits = _limits(
            problem,
            cfg.padding_ratio,
            square=True,
        )
        ax.set_facecolor("white")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        return fig, ax, header_ax, side_ax, style

    # Legacy/non-fixed rendering path.
    has_header = _needs_header(problem, demand_encoding)
    has_side_panel = _uses_side_panel(demand_encoding)
    figure_height = style.figure_size_inches + (0.85 if has_header else 0.0)
    figure_width = style.figure_size_inches + (
        demand_encoding.side_panel_width_inches if has_side_panel else 0.0
    )
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=cfg.dpi)
    fig.patch.set_facecolor("white")

    if has_side_panel:
        if has_header:
            grid = fig.add_gridspec(
                nrows=2,
                ncols=2,
                height_ratios=(0.85, style.figure_size_inches),
                width_ratios=(
                    style.figure_size_inches,
                    demand_encoding.side_panel_width_inches,
                ),
                hspace=0.03,
                wspace=0.03,
            )
            header_ax = fig.add_subplot(grid[0, :])
            ax = fig.add_subplot(grid[1, 0])
            side_ax = fig.add_subplot(grid[1, 1])
        else:
            grid = fig.add_gridspec(
                nrows=1,
                ncols=2,
                width_ratios=(
                    style.figure_size_inches,
                    demand_encoding.side_panel_width_inches,
                ),
                wspace=0.03,
            )
            header_ax = None
            ax = fig.add_subplot(grid[0, 0])
            side_ax = fig.add_subplot(grid[0, 1])
    else:
        side_ax = None
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

    if side_ax is not None:
        side_ax.set_facecolor("white")
        side_ax.set_xlim(0.0, 1.0)
        side_ax.set_ylim(0.0, 1.0)
        side_ax.axis("off")
        side_ax.plot(
            [0.02, 0.02],
            [0.02, 0.98],
            color="#D1D5DB",
            linewidth=1.0,
            transform=side_ax.transAxes,
        )

    return fig, ax, header_ax, side_ax, style

def _demand_ratio(problem: ProblemInstance, node: int) -> float:
    if node == problem.depot:
        return 0.0
    demand = problem.demands.get(node, 0)
    return min(1.0, max(0.0, demand / problem.capacity))

def _node_radius_data(style: _RenderStyle, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> float:
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    usable_axis_points = style.figure_size_inches * 72.0 * 0.78
    data_per_point = min(x_span, y_span) / max(usable_axis_points, 1.0)
    scatter_radius_points = math.sqrt(style.base_node_size) * 0.50
    return max(data_per_point * scatter_radius_points, min(x_span, y_span) * 0.003)

def _data_per_point(style: _RenderStyle, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> float:
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    usable_axis_points = style.figure_size_inches * 72.0 * 0.78
    return min(x_span, y_span) / max(usable_axis_points, 1.0)

def _label_color(fill_color: str) -> str:
    red, green, blue = to_rgb(fill_color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.45 else "#111827"

def _demand_colormap(demand_encoding: DemandEncodingConfig) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("avma_cvrp_demand", list(demand_encoding.color_stops))

def _draw_size_reference(ax: Axes, demand_encoding: DemandEncodingConfig, style: _RenderStyle) -> None:
    small_area = style.base_node_size * demand_encoding.size_min_factor
    full_area = style.base_node_size * demand_encoding.size_max_factor
    ax.scatter([0.40, 0.56], [0.50, 0.50], s=[small_area, full_area], facecolors=DEMAND_FILL_COLOR, edgecolors=NODE_EDGE_COLOR, linewidths=1.3, transform=ax.transAxes, zorder=2)

def _bar_reference_dimensions(
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> tuple[float, float]:
    # Keep the reference bar physically consistent with the map bars.
    return (
        float(demand_encoding.bar_width_points) * style.bar_scale,
        float(demand_encoding.bar_height_points) * style.bar_scale,
    )

def _draw_bar_reference(
    ax: Axes,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> None:
    width_points, height_points = _bar_reference_dimensions(
        demand_encoding,
        style,
    )

    # Centered header slots: empty reference | full reference | vehicles.
    for center_x, full in ((0.32, False), (0.50, True)):
        drawing = DrawingArea(width_points, height_points, 0, 0, clip=False)
        drawing.add_artist(
            Rectangle(
                (0.0, 0.0),
                width_points,
                height_points,
                facecolor=NODE_FILL_COLOR,
                edgecolor=NODE_EDGE_COLOR,
                linewidth=1.0,
            )
        )
        if full:
            drawing.add_artist(
                Rectangle(
                    (0.0, 0.0),
                    width_points,
                    height_points,
                    facecolor=DEMAND_FILL_COLOR,
                    edgecolor="none",
                )
            )
        ax.add_artist(
            AnnotationBbox(
                drawing,
                (center_x, 0.50),
                xycoords=ax.transAxes,
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0.0,
                zorder=3,
            )
        )

def _dot_positions(*, node: int, ratio: float, radius: float, grid_size: int) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    denominator = max(grid_size - 1, 1)
    for row in range(grid_size):
        for column in range(grid_size):
            relative_x = -0.78 + 1.56 * column / denominator
            relative_y = -0.78 + 1.56 * row / denominator
            if math.hypot(relative_x, relative_y) <= 0.88:
                candidates.append((relative_x * radius, relative_y * radius))
    generator = random.Random(f"avma-cvrp-dot-density-{node}")
    generator.shuffle(candidates)
    count = 0 if ratio <= 0.0 else max(1, int(round(ratio * len(candidates))))
    return candidates[:count]

def _dot_safe_placement_radius_points(
    marker_radius_points: float,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> float:
    # Keep dot centers inside the visible circle after accounting for
    # the dot radius and half of the circle stroke width.
    rendered_dot_radius_points = max(
        1.0,
        float(demand_encoding.dot_radius_points),
    )
    safe_center_radius = max(
        0.0,
        marker_radius_points
        - rendered_dot_radius_points
        - style.node_line_width * 0.50,
    )

    # _dot_positions only accepts candidate centers inside normalized
    # radius 0.88. Preserve the candidate set/count and only inset it.
    return safe_center_radius / 0.88

def _draw_dot_reference(ax: Axes, demand_encoding: DemandEncodingConfig, style: _RenderStyle) -> None:
    diameter_points = math.sqrt(style.base_node_size)
    padding_points = 1.5
    drawing_size = diameter_points + 2.0 * padding_points
    marker_radius = diameter_points * 0.50
    for center_x, ratio in ((0.40, 0.0), (0.58, 1.0)):
        drawing = DrawingArea(drawing_size, drawing_size, 0, 0, clip=False)
        center = padding_points + marker_radius
        drawing.add_artist(Circle((center, center), marker_radius, facecolor=NODE_FILL_COLOR, edgecolor=NODE_EDGE_COLOR, linewidth=style.node_line_width))
        for offset_x, offset_y in _dot_positions(node=0, ratio=ratio, radius=_dot_safe_placement_radius_points(marker_radius, demand_encoding, style), grid_size=demand_encoding.dot_grid_size):
            drawing.add_artist(Circle((center + offset_x, center + offset_y), demand_encoding.dot_radius_points, facecolor=NODE_EDGE_COLOR, edgecolor="none"))
        ax.add_artist(AnnotationBbox(drawing, (center_x, 0.50), xycoords=ax.transAxes, frameon=False, box_alignment=(0.5, 0.5), pad=0.0, zorder=3))

def _draw_color_reference(ax: Axes, demand_encoding: DemandEncodingConfig, style: _RenderStyle) -> None:
    color_map = _demand_colormap(demand_encoding)
    ax.scatter([0.40, 0.58], [0.50, 0.50], s=[style.base_node_size, style.base_node_size], facecolors=[color_map(0.0), color_map(1.0)], edgecolors=NODE_EDGE_COLOR, linewidths=1.2, transform=ax.transAxes, zorder=2)

def _draw_visual_demand_legend(
    ax: Axes,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> None:
    if demand_encoding.mode == "size":
        _draw_size_reference(ax, demand_encoding, style)
    elif demand_encoding.mode == "bar":
        _draw_bar_reference(ax, demand_encoding, style)
    elif demand_encoding.mode == "dot_density":
        _draw_dot_reference(ax, demand_encoding, style)
    elif demand_encoding.mode == "color":
        _draw_color_reference(ax, demand_encoding, style)

def _draw_vehicle_icons(ax: Axes, max_vehicles: int, style: _RenderStyle) -> None:
    if max_vehicles <= 0:
        return

    width = style.vehicle_icon_width_points
    height = style.vehicle_icon_height_points
    drawing = DrawingArea(width, height, 0, 0, clip=False)

    body_y = height * 0.28
    body_h = height * 0.38
    cargo_w = width * 0.52
    cab_w = width * 0.22

    drawing.add_artist(
        Rectangle(
            (0.0, body_y),
            cargo_w,
            body_h,
            facecolor="#D1D5DB",
            edgecolor="#374151",
            linewidth=0.9,
        )
    )
    drawing.add_artist(
        Rectangle(
            (cargo_w, body_y + body_h * 0.10),
            cab_w,
            body_h * 0.90,
            facecolor="#9CA3AF",
            edgecolor="#374151",
            linewidth=0.9,
        )
    )

    wheel_radius = height * 0.15
    for wheel_x in (width * 0.18, width * 0.57):
        drawing.add_artist(
            Circle(
                (wheel_x, body_y - wheel_radius * 0.05),
                wheel_radius,
                facecolor="#111827",
                edgecolor="none",
            )
        )

    ax.add_artist(
        AnnotationBbox(
            drawing,
            (0.66, 0.50),
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0.0,
            zorder=3,
        )
    )

    ax.text(
        0.70,
        0.50,
        f"× {int(max_vehicles)}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=style.vehicle_text_size,
        color="#111827",
        weight="bold",
        zorder=4,
    )

def _draw_header(ax: Axes | None, problem: ProblemInstance, demand_encoding: DemandEncodingConfig, style: _RenderStyle) -> None:
    if ax is None:
        return
    if demand_encoding.mode != "none" and demand_encoding.show_visual_legend:
        _draw_visual_demand_legend(ax, demand_encoding, style)
    if demand_encoding.show_vehicle_icons and problem.max_vehicles is not None:
        _draw_vehicle_icons(ax, problem.max_vehicles, style)

def _bar_dimensions_data(
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[float, float]:
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    usable_axis_points = style.figure_size_inches * 72.0 * 0.78

    width_points = (
        demand_encoding.bar_width_points
        * style.bar_scale
    )
    height_points = (
        demand_encoding.bar_height_points
        * style.bar_scale
    )

    return (
        x_span * width_points / usable_axis_points,
        y_span * height_points / usable_axis_points,
    )

def _local_bar_rect(x: float, y: float, style: _RenderStyle, demand_encoding: DemandEncodingConfig, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> Rect:
    width, height = _bar_dimensions_data(style, demand_encoding, x_limits, y_limits)
    data_per_point = _data_per_point(style, x_limits, y_limits)
    bar_node_area = max(18.0, style.base_node_size * 0.08)
    node_radius = data_per_point * math.sqrt(bar_node_area) * 0.50
    return x - width / 2.0, max(y_limits[0], y - node_radius - height * 1.25), width, height

def _draw_bar_rect(ax: Axes, rect: Rect, ratio: float) -> None:
    left, bottom, width, height = rect
    ax.add_patch(Rectangle((left, bottom), width, height, facecolor=NODE_FILL_COLOR, edgecolor=NODE_EDGE_COLOR, linewidth=1.0, zorder=4))
    ax.add_patch(Rectangle((left, bottom), width * ratio, height, facecolor=DEMAND_FILL_COLOR, edgecolor="none", zorder=4.1))

def _rect_overlap_area(first: Rect, second: Rect) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap_x = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_y = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return overlap_x * overlap_y

def _label_size_data(
    node: int,
    style: _RenderStyle,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[float, float]:
    dpp = _data_per_point(style, x_limits, y_limits)
    width = max(1, len(str(node))) * style.font_size * 0.62 * dpp
    height = style.font_size * 1.05 * dpp
    return width, height


def _centered_rect(
    cx: float,
    cy: float,
    width: float,
    height: float,
) -> Rect:
    return cx - width / 2.0, cy - height / 2.0, width, height


def _label_marker_radius(
    problem: ProblemInstance,
    node: int,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> float:
    dpp = _data_per_point(style, x_limits, y_limits)

    if demand_encoding.mode == "bar":
        marker_area = max(18.0, style.base_node_size * 0.08)
    elif demand_encoding.mode == "size":
        ratio = _demand_ratio(problem, node)
        scale = demand_encoding.size_min_factor + (
            demand_encoding.size_max_factor
            - demand_encoding.size_min_factor
        ) * ratio
        marker_area = style.base_node_size * scale
    else:
        marker_area = style.base_node_size

    return dpp * math.sqrt(marker_area) * 0.50


def _candidate_label_rects(
    problem: ProblemInstance,
    node: int,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> list[tuple[float, float, Rect, int, int]]:
    x, y = problem.coordinates[node]
    dpp = _data_per_point(style, x_limits, y_limits)
    width, height = _label_size_data(node, style, x_limits, y_limits)
    marker_radius = _label_marker_radius(
        problem,
        node,
        style,
        demand_encoding,
        x_limits,
        y_limits,
    )

    # Genel olarak node'a daha yakin; temas etmemesi icin kucuk bir bosluk.
    clearance = max(1.15 * dpp, marker_radius * 0.16) * style.label_clearance_scale

    directions = (
        (0.70710678, 0.70710678),    # NE
        (0.0, 1.0),                  # N
        (1.0, 0.0),                  # E
        (-0.70710678, 0.70710678),   # NW
        (-1.0, 0.0),                 # W
        (0.70710678, -0.70710678),   # SE
        (0.0, -1.0),                 # S
        (-0.70710678, -0.70710678),  # SW
    )

    ring_step = 1.75 * style.label_clearance_scale
    extra_rings_points = (0.0, ring_step, ring_step * 2.0, ring_step * 3.0, ring_step * 4.0)

    candidates: list[tuple[float, float, Rect, int, int]] = []
    for ring_index, extra_points in enumerate(extra_rings_points):
        extra = extra_points * dpp
        for direction_index, (ux, uy) in enumerate(directions):
            cx = x + ux * (
                marker_radius + clearance + width / 2.0 + extra
            )
            cy = y + uy * (
                marker_radius + clearance + height / 2.0 + extra
            )
            rect = _centered_rect(cx, cy, width, height)
            candidates.append(
                (cx, cy, rect, ring_index, direction_index)
            )

    return candidates



def _node_rect(x: float, y: float, style: _RenderStyle, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> Rect:
    radius = _node_radius_data(style, x_limits, y_limits) * 0.70
    return x - radius, y - radius, radius * 2.0, radius * 2.0

def _depot_rect(x: float, y: float, style: _RenderStyle, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> Rect:
    """Occupied depot region used only by collision-aware bar placement."""
    dpp = _data_per_point(style, x_limits, y_limits)
    half_side = math.sqrt(style.depot_size) * 0.50 * dpp
    # Keep a small visual clearance around the square depot marker as well.
    clearance = max(2.0 * dpp, half_side * 0.25)
    half_extent = half_side + clearance
    return x - half_extent, y - half_extent, half_extent * 2.0, half_extent * 2.0

def _candidate_bar_rects(x: float, y: float, style: _RenderStyle, demand_encoding: DemandEncodingConfig, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> list[Rect]:
    width, height = _bar_dimensions_data(style, demand_encoding, x_limits, y_limits)
    dpp = _data_per_point(style, x_limits, y_limits)
    bar_node_area = max(18.0, style.base_node_size * 0.08)
    node_radius = dpp * math.sqrt(bar_node_area) * 0.50
    gap = max(height * 0.45, dpp * 2.0)
    vertical = node_radius + gap + height / 2.0
    horizontal = node_radius + gap + width / 2.0
    centers = [
        (x, y - vertical), (x, y + vertical), (x + horizontal, y), (x - horizontal, y),
        (x + horizontal, y - vertical), (x - horizontal, y - vertical),
        (x + horizontal, y + vertical), (x - horizontal, y + vertical),
    ]
    return [(cx - width / 2.0, cy - height / 2.0, width, height) for cx, cy in centers]

def _outside_penalty(rect: Rect, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> float:
    left, bottom, width, height = rect
    right, top = left + width, bottom + height
    overflow = max(0.0, x_limits[0] - left) + max(0.0, right - x_limits[1]) + max(0.0, y_limits[0] - bottom) + max(0.0, top - y_limits[1])
    return overflow * 1_000_000.0

def _collision_bar_positions(
    problem: ProblemInstance,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> dict[int, Rect]:
    customers = sorted(
        node for node in problem.node_ids if node != problem.depot
    )
    node_rects = {
        node: _node_rect(
            *problem.coordinates[node],
            style,
            x_limits,
            y_limits,
        )
        for node in customers
    }
    depot_rect = _depot_rect(
        *problem.coordinates[problem.depot],
        style,
        x_limits,
        y_limits,
    )

    chosen: dict[int, Rect] = {}

    for node in customers:
        candidates = _candidate_bar_rects(
            *problem.coordinates[node],
            style,
            demand_encoding,
            x_limits,
            y_limits,
        )
        best_index = 0
        best_score = math.inf

        for index, rect in enumerate(candidates):
            score = _outside_penalty(rect, x_limits, y_limits)

            depot_overlap = _rect_overlap_area(rect, depot_rect)
            if depot_overlap > 0.0:
                score += (
                    1_000_000_000.0
                    + depot_overlap * 1_000_000.0
                )

            for other, other_rect in node_rects.items():
                if other != node:
                    score += (
                        _rect_overlap_area(rect, other_rect) * 25.0
                    )

            for bar_rect in chosen.values():
                score += (
                    _rect_overlap_area(rect, bar_rect) * 35.0
                )

            score += index * 1e-9

            if score < best_score:
                best_score = score
                best_index = index

        chosen[node] = candidates[best_index]

    return chosen


def _label_processing_order(
    problem: ProblemInstance,
) -> list[int]:
    customers = sorted(
        node for node in problem.node_ids if node != problem.depot
    )
    all_nodes = tuple(problem.node_ids)

    def nearest_distance(node: int) -> float:
        x, y = problem.coordinates[node]
        best = math.inf
        for other in all_nodes:
            if other == node:
                continue
            ox, oy = problem.coordinates[other]
            best = min(best, math.hypot(ox - x, oy - y))
        return best

    return sorted(
        customers,
        key=lambda node: (nearest_distance(node), node),
    )


def _collision_label_positions(
    problem: ProblemInstance,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    map_bar_rects: dict[int, Rect] | None = None,
) -> dict[int, tuple[float, float]]:
    customers = sorted(
        node for node in problem.node_ids if node != problem.depot
    )
    node_rects = {
        node: _node_rect(
            *problem.coordinates[node],
            style,
            x_limits,
            y_limits,
        )
        for node in customers
    }
    depot_rect = _depot_rect(
        *problem.coordinates[problem.depot],
        style,
        x_limits,
        y_limits,
    )
    bars = map_bar_rects or {}

    chosen_positions: dict[int, tuple[float, float]] = {}
    chosen_rects: dict[int, Rect] = {}

    for node in _label_processing_order(problem):
        best_score = math.inf
        best_position: tuple[float, float] | None = None
        best_rect: Rect | None = None

        candidates = _candidate_label_rects(
            problem,
            node,
            style,
            demand_encoding,
            x_limits,
            y_limits,
        )

        for (
            cx,
            cy,
            rect,
            ring_index,
            direction_index,
        ) in candidates:
            outside = _outside_penalty(
                rect,
                x_limits,
                y_limits,
            )
            score = 0.0

            if outside > 0.0:
                score += 10_000_000_000.0 + outside

            depot_overlap = _rect_overlap_area(
                rect,
                depot_rect,
            )
            if depot_overlap > 0.0:
                score += (
                    1_000_000_000.0
                    + depot_overlap * 1_000_000.0
                )

            for other, other_rect in node_rects.items():
                if other == node:
                    continue
                overlap = _rect_overlap_area(
                    rect,
                    other_rect,
                )
                if overlap > 0.0:
                    score += (
                        100_000_000.0
                        + overlap * 1_000_000.0
                    )

            for other_rect in chosen_rects.values():
                overlap = _rect_overlap_area(
                    rect,
                    other_rect,
                )
                if overlap > 0.0:
                    score += (
                        1_000_000_000.0
                        + overlap * 1_000_000.0
                    )

            for bar_rect in bars.values():
                overlap = _rect_overlap_area(
                    rect,
                    bar_rect,
                )
                if overlap > 0.0:
                    score += (
                        100_000_000.0
                        + overlap * 1_000_000.0
                    )

            score += (
                ring_index * 10.0
                + direction_index * 0.01
            )

            if score < best_score:
                best_score = score
                best_position = (cx, cy)
                best_rect = rect

        if best_position is None or best_rect is None:
            raise RuntimeError(
                f"Label placement failed for node {node}"
            )

        chosen_positions[node] = best_position
        chosen_rects[node] = best_rect

    return chosen_positions



def _draw_dot_density_marker(ax: Axes, *, node: int, x: float, y: float, ratio: float, style: _RenderStyle, demand_encoding: DemandEncodingConfig, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    marker_radius_points = math.sqrt(style.base_node_size) * 0.50
    placement_radius_points = _dot_safe_placement_radius_points(
        marker_radius_points,
        demand_encoding,
        style,
    )

    bbox = ax.get_window_extent()
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    data_per_point_x = (
        x_span / max(float(bbox.width), 1.0)
    ) * ax.figure.dpi / 72.0
    data_per_point_y = (
        y_span / max(float(bbox.height), 1.0)
    ) * ax.figure.dpi / 72.0
    radius = placement_radius_points * min(
        data_per_point_x,
        data_per_point_y,
    )
    offsets = _dot_positions(node=node, ratio=ratio, radius=radius, grid_size=demand_encoding.dot_grid_size)
    if not offsets:
        return
    xs = [x + offset_x for offset_x, _ in offsets]
    ys = [y + offset_y for _, offset_y in offsets]
    dot_area = max(4.0, (demand_encoding.dot_radius_points * 2.0) ** 2)
    ax.scatter(xs, ys, s=dot_area, facecolors=NODE_EDGE_COLOR, edgecolors="none", zorder=4)

def _draw_customer_node(
    ax: Axes,
    problem: ProblemInstance,
    node: int,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    color_map: LinearSegmentedColormap | None,
    collision_bars: dict[int, Rect] | None,
    label_positions: dict[int, tuple[float, float]],
) -> None:
    x, y = problem.coordinates[node]
    ratio = _demand_ratio(problem, node)

    marker_area = style.base_node_size
    fill_color = NODE_FILL_COLOR
    marker_edge_color = NODE_EDGE_COLOR
    marker_line_width = style.node_line_width

    if demand_encoding.mode == "size":
        scale = demand_encoding.size_min_factor + (
            demand_encoding.size_max_factor
            - demand_encoding.size_min_factor
        ) * ratio
        marker_area = style.base_node_size * scale
        fill_color = DEMAND_FILL_COLOR

    elif (
        demand_encoding.mode == "color"
        and color_map is not None
    ):
        fill_color = color_map(ratio)

    elif demand_encoding.mode == "bar":
        marker_area = max(
            18.0,
            style.base_node_size * 0.08,
        )
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

    if (
        demand_encoding.mode == "bar"
        and demand_encoding.bar_layout != "side_panel"
    ):
        if (
            demand_encoding.bar_layout == "collision_aware"
            and collision_bars is not None
        ):
            rect = collision_bars[node]
        else:
            rect = _local_bar_rect(
                x,
                y,
                style,
                demand_encoding,
                x_limits,
                y_limits,
            )
        _draw_bar_rect(ax, rect, ratio)

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

    label_x, label_y = label_positions[node]
    label = ax.text(
        label_x,
        label_y,
        str(node),
        ha="center",
        va="center",
        fontsize=style.font_size,
        color="#111827",
        weight="bold",
        zorder=6,
    )
    label.set_path_effects(
        [
            path_effects.withStroke(
                linewidth=2.0,
                foreground="white",
            )
        ]
    )



def _draw_depot(ax: Axes, problem: ProblemInstance, style: _RenderStyle) -> None:
    x, y = problem.coordinates[problem.depot]
    ax.scatter([x], [y], s=style.depot_size, marker="s", facecolors=DEPOT_FILL_COLOR, edgecolors="#000000", linewidths=style.depot_line_width, zorder=4)
    ax.text(x, y, str(problem.depot), ha="center", va="center", fontsize=style.font_size, color="white", weight="bold", zorder=5)

def _draw_nodes(
    ax: Axes,
    problem: ProblemInstance,
    style: _RenderStyle,
    demand_encoding: DemandEncodingConfig,
) -> None:
    # _8method_v3_neutral_side_map
    if _uses_side_panel(demand_encoding) and demand_encoding.mode != "bar":
        demand_encoding = replace(
            demand_encoding,
            mode="bar",
            bar_layout="side_panel",
        )
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    color_map = (
        _demand_colormap(demand_encoding)
        if demand_encoding.mode == "color"
        else None
    )

    collision_bars: dict[int, Rect] | None = None
    map_bar_rects: dict[int, Rect] = {}

    if (
        demand_encoding.mode == "bar"
        and demand_encoding.bar_layout == "collision_aware"
    ):
        collision_bars = _collision_bar_positions(
            problem,
            style,
            demand_encoding,
            x_limits,
            y_limits,
        )
        map_bar_rects = collision_bars

    elif (
        demand_encoding.mode == "bar"
        and demand_encoding.bar_layout == "local"
    ):
        map_bar_rects = {
            node: _local_bar_rect(
                *problem.coordinates[node],
                style,
                demand_encoding,
                x_limits,
                y_limits,
            )
            for node in problem.node_ids
            if node != problem.depot
        }

    label_positions = _collision_label_positions(
        problem,
        style,
        demand_encoding,
        x_limits,
        y_limits,
        map_bar_rects=map_bar_rects,
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
            collision_bars,
            label_positions,
        )

    _draw_depot(ax, problem, style)



def _side_bar_drawing(
    ratio: float,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> DrawingArea:
    width = (
        float(demand_encoding.side_bar_width_points)
        * style.bar_scale
    )
    height = (
        float(demand_encoding.side_bar_height_points)
        * style.bar_scale
    )
    drawing = DrawingArea(width, height, 0, 0, clip=False)
    drawing.add_artist(
        Rectangle(
            (0.0, 0.0),
            width,
            height,
            facecolor=NODE_FILL_COLOR,
            edgecolor=NODE_EDGE_COLOR,
            linewidth=1.0,
        )
    )
    drawing.add_artist(
        Rectangle(
            (0.0, 0.0),
            width * ratio,
            height,
            facecolor=DEMAND_FILL_COLOR,
            edgecolor="none",
        )
    )
    return drawing

def _side_glyph_dimensions_points(
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> tuple[float, float]:
    if demand_encoding.mode == "bar":
        return (
            float(demand_encoding.side_bar_width_points) * style.bar_scale,
            float(demand_encoding.side_bar_height_points) * style.bar_scale,
        )
    if demand_encoding.mode == "size":
        diameter = math.sqrt(
            style.base_node_size * demand_encoding.size_max_factor
        )
        return diameter, diameter
    if demand_encoding.mode in {"color", "dot_density"}:
        diameter = math.sqrt(style.base_node_size)
        return diameter, diameter
    raise ValueError(
        f"Unsupported side-panel demand encoding: {demand_encoding.mode}"
    )


def _side_size_drawing(
    ratio: float,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> DrawingArea:
    scale = (
        demand_encoding.size_min_factor
        + (
            demand_encoding.size_max_factor
            - demand_encoding.size_min_factor
        )
        * ratio
    )
    diameter = math.sqrt(style.base_node_size * scale)
    padding = 1.5
    drawing = DrawingArea(
        diameter + 2.0 * padding,
        diameter + 2.0 * padding,
        0,
        0,
        clip=False,
    )
    center = padding + diameter * 0.5
    drawing.add_artist(
        Circle(
            (center, center),
            diameter * 0.5,
            facecolor=DEMAND_FILL_COLOR,
            edgecolor=NODE_EDGE_COLOR,
            linewidth=style.node_line_width,
        )
    )
    return drawing


def _side_color_drawing(
    ratio: float,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> DrawingArea:
    diameter = math.sqrt(style.base_node_size)
    padding = 1.5
    drawing = DrawingArea(
        diameter + 2.0 * padding,
        diameter + 2.0 * padding,
        0,
        0,
        clip=False,
    )
    center = padding + diameter * 0.5
    drawing.add_artist(
        Circle(
            (center, center),
            diameter * 0.5,
            facecolor=_demand_colormap(demand_encoding)(ratio),
            edgecolor=NODE_EDGE_COLOR,
            linewidth=style.node_line_width,
        )
    )
    return drawing


def _side_dot_density_drawing(
    node: int,
    ratio: float,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> DrawingArea:
    diameter = math.sqrt(style.base_node_size)
    padding = 1.5
    drawing_size = diameter + 2.0 * padding
    marker_radius = diameter * 0.5
    drawing = DrawingArea(
        drawing_size,
        drawing_size,
        0,
        0,
        clip=False,
    )
    center = padding + marker_radius
    drawing.add_artist(
        Circle(
            (center, center),
            marker_radius,
            facecolor=NODE_FILL_COLOR,
            edgecolor=NODE_EDGE_COLOR,
            linewidth=style.node_line_width,
        )
    )
    for offset_x, offset_y in _dot_positions(
        node=node,
        ratio=ratio,
        radius=_dot_safe_placement_radius_points(
            marker_radius,
            demand_encoding,
            style,
        ),
        grid_size=demand_encoding.dot_grid_size,
    ):
        drawing.add_artist(
            Circle(
                (center + offset_x, center + offset_y),
                demand_encoding.dot_radius_points,
                facecolor=NODE_EDGE_COLOR,
                edgecolor="none",
            )
        )
    return drawing


def _side_glyph_drawing(
    node: int,
    ratio: float,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
) -> DrawingArea:
    if demand_encoding.mode == "bar":
        return _side_bar_drawing(ratio, demand_encoding, style)
    if demand_encoding.mode == "size":
        return _side_size_drawing(ratio, demand_encoding, style)
    if demand_encoding.mode == "color":
        return _side_color_drawing(ratio, demand_encoding, style)
    if demand_encoding.mode == "dot_density":
        return _side_dot_density_drawing(
            node,
            ratio,
            demand_encoding,
            style,
        )
    raise ValueError(
        f"Unsupported side-panel demand encoding: {demand_encoding.mode}"
    )

def _side_panel_layout_metrics(
    problem: ProblemInstance,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
    cfg: RenderConfig,
) -> dict[str, int | bool]:
    customers = sorted(
        node for node in problem.node_ids if node != problem.depot
    )
    count = len(customers)
    if count == 0:
        return {
            "columns": 0,
            "rows": 0,
            "max_columns_by_width": 0,
            "max_rows_by_height": 0,
            "over_capacity": False,
        }

    if not cfg.fixed_canvas:
        columns = min(
            max(1, demand_encoding.side_panel_columns or 1),
            count,
        )
        rows = math.ceil(count / columns)
        return {
            "columns": columns,
            "rows": rows,
            "max_columns_by_width": columns,
            "max_rows_by_height": rows,
            "over_capacity": False,
        }

    inner_width_px = cfg.panel_width_px * 0.96
    inner_height_px = cfg.canvas_height_px * 0.96
    font_px = style.font_size * cfg.dpi / 72.0
    max_digits = max(len(str(node)) for node in customers)
    id_width_px = max_digits * font_px * 0.62

    glyph_width_points, glyph_height_points = _side_glyph_dimensions_points(
        demand_encoding,
        style,
    )
    glyph_width_px = glyph_width_points * cfg.dpi / 72.0
    glyph_height_px = glyph_height_points * cfg.dpi / 72.0

    required_column_px = (
        id_width_px
        + 8.0
        + glyph_width_px
        + 12.0
    )
    max_columns_by_width = max(
        1,
        int(inner_width_px // required_column_px),
    )

    required_row_px = (
        max(font_px * 0.78, glyph_height_px * 1.05)
        + 2.0
    )
    max_rows_by_height = max(
        1,
        int(inner_height_px // required_row_px),
    )
    needed_columns = max(
        1,
        math.ceil(count / max_rows_by_height),
    )

    columns = min(max_columns_by_width, needed_columns)
    rows = math.ceil(count / columns)
    over_capacity = rows > max_rows_by_height

    return {
        "columns": columns,
        "rows": rows,
        "max_columns_by_width": max_columns_by_width,
        "max_rows_by_height": max_rows_by_height,
        "over_capacity": over_capacity,
    }


def _draw_side_panel(
    side_ax: Axes | None,
    problem: ProblemInstance,
    demand_encoding: DemandEncodingConfig,
    style: _RenderStyle,
    cfg: RenderConfig,
) -> None:
    if side_ax is None or not _uses_side_panel(demand_encoding):
        return

    customers = sorted(
        node for node in problem.node_ids if node != problem.depot
    )
    if not customers:
        return

    metrics = _side_panel_layout_metrics(
        problem,
        demand_encoding,
        style,
        cfg,
    )
    columns = int(metrics["columns"])
    rows = int(metrics["rows"])

    left = 0.035
    right = 0.985
    top = 0.985
    bottom = 0.015
    usable_width = right - left
    usable_height = top - bottom
    column_width = usable_width / columns

    for index, node in enumerate(customers):
        column = min(index // rows, columns - 1)
        row = index % rows
        x0 = left + column * column_width
        y = top - (row + 0.5) * usable_height / rows
        id_x = x0 + column_width * 0.26
        glyph_x = x0 + column_width * 0.68

        side_ax.text(
            id_x,
            y,
            str(node),
            transform=side_ax.transAxes,
            ha="right",
            va="center",
            fontsize=style.font_size,
            color="#111827",
            weight="bold",
            zorder=5,
        )
        side_ax.add_artist(
            AnnotationBbox(
                _side_glyph_drawing(
                    node,
                    _demand_ratio(problem, node),
                    demand_encoding,
                    style,
                ),
                (glyph_x, y),
                xycoords=side_ax.transAxes,
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0.0,
                zorder=4,
            )
        )


def _draw_routes(ax: Axes, problem: ProblemInstance, routes: Sequence[Iterable[int]], style: _RenderStyle, route_rendering: RouteRenderingConfig) -> None:
    routes_tuple = tuple(tuple(route) for route in routes)
    palette = route_rendering.palette
    for route_index, route in enumerate(routes_tuple):
        color = palette[route_index % len(palette)]
        for first, second in zip(route, route[1:]):
            ax.plot([problem.coordinates[first][0], problem.coordinates[second][0]], [problem.coordinates[first][1], problem.coordinates[second][1]], color=color, linewidth=style.route_line_width, zorder=1)

def _validate_renderable_nodes(problem: ProblemInstance, routes: Sequence[Iterable[int]]) -> None:
    unknown_nodes = sorted({node for route in routes for node in route if node not in problem.coordinates})
    if unknown_nodes:
        raise ValueError(f"Render edilemeyen bilinmeyen node ID'leri: {unknown_nodes}")

def _save_figure(
    fig: plt.Figure,
    output_path: Path,
    cfg: RenderConfig,
    *,
    pad_inches: float,
) -> None:
    if cfg.fixed_canvas:
        width_in, height_in = fig.get_size_inches()
        expected = (
            int(round(width_in * cfg.dpi)),
            int(round(height_in * cfg.dpi)),
        )

        fig.savefig(
            output_path,
            dpi=cfg.dpi,
            bbox_inches=None,
            pad_inches=0.0,
            facecolor="white",
        )

        with Image.open(output_path) as image:
            actual = image.size

        if actual != expected:
            raise RuntimeError(
                f"Frozen canvas boyutu bozuldu: expected={expected}, actual={actual}"
            )
    else:
        fig.savefig(output_path, bbox_inches="tight", pad_inches=pad_inches)

def _write_render_metadata(
    output_path: Path,
    cfg: RenderConfig,
    demand_encoding: DemandEncodingConfig,
    problem: ProblemInstance,
    style: _RenderStyle,
) -> Path:
    with Image.open(output_path) as image:
        width_px, height_px = image.size
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    side_metrics = None
    if _uses_side_panel(demand_encoding):
        side_metrics = _side_panel_layout_metrics(
            problem,
            demand_encoding,
            style,
            cfg,
        )

    header_height_px = None
    map_viewport_px = None
    if cfg.fixed_canvas:
        header_height_px = min(
            max(int(cfg.panel_header_height_px), 0),
            max(int(cfg.canvas_height_px) - 1, 0),
        )
        map_viewport_px = min(
            int(cfg.map_width_px),
            int(cfg.canvas_height_px) - header_height_px,
        )

    customer_count = sum(
        1 for node in problem.node_ids if node != problem.depot
    )

    metadata = {
        "image": output_path.name,
        "width_px": width_px,
        "height_px": height_px,
        "sha256": digest,
        "dpi": cfg.dpi,
        "fixed_canvas": cfg.fixed_canvas,
        "canvas_width_px": width_px,
        "canvas_height_px": height_px,
        "map_width_px": map_viewport_px,
        "map_height_px": map_viewport_px,
        "header_height_px": header_height_px,
        "panel_width_px": (
            cfg.panel_width_px
            if cfg.fixed_canvas and _uses_side_panel(demand_encoding)
            else None
        ),
        "panel_header_height_px": (
            0
            if cfg.fixed_canvas and _uses_side_panel(demand_encoding)
            else None
        ),
        "padding_ratio": cfg.padding_ratio,
        "customer_count": customer_count,
        "primitive_scale": style.primitive_scale,
        "font_size_points": style.font_size,
        "node_marker_area_points2": style.base_node_size,
        "depot_marker_area_points2": style.depot_size,
        "route_line_width_points": style.route_line_width,
        "demand_encoding_mode": demand_encoding.mode,
        "demand_placement": _demand_placement(demand_encoding),
        "bar_layout": (
            demand_encoding.bar_layout
            if demand_encoding.mode == "bar"
            else None
        ),
        "bar_width_points_configured": (
            demand_encoding.bar_width_points
            if demand_encoding.mode == "bar"
            else None
        ),
        "bar_height_points_configured": (
            demand_encoding.bar_height_points
            if demand_encoding.mode == "bar"
            else None
        ),
        "bar_width_points_actual": (
            demand_encoding.bar_width_points * style.bar_scale
            if demand_encoding.mode == "bar"
            else None
        ),
        "bar_height_points_actual": (
            demand_encoding.bar_height_points * style.bar_scale
            if demand_encoding.mode == "bar"
            else None
        ),
        "side_bar_width_points_configured": (
            demand_encoding.side_bar_width_points
            if _uses_side_panel(demand_encoding)
            else None
        ),
        "side_bar_height_points_configured": (
            demand_encoding.side_bar_height_points
            if _uses_side_panel(demand_encoding)
            else None
        ),
        "side_bar_width_points_actual": (
            demand_encoding.side_bar_width_points * style.bar_scale
            if _uses_side_panel(demand_encoding)
            else None
        ),
        "side_bar_height_points_actual": (
            demand_encoding.side_bar_height_points * style.bar_scale
            if _uses_side_panel(demand_encoding)
            else None
        ),
        "side_panel_columns_actual": (
            int(side_metrics["columns"])
            if side_metrics is not None
            else None
        ),
        "side_panel_rows_actual": (
            int(side_metrics["rows"])
            if side_metrics is not None
            else None
        ),
        "side_panel_max_columns_by_width": (
            int(side_metrics["max_columns_by_width"])
            if side_metrics is not None
            else None
        ),
        "side_panel_max_rows_by_height": (
            int(side_metrics["max_rows_by_height"])
            if side_metrics is not None
            else None
        ),
        "side_panel_over_capacity": (
            bool(side_metrics["over_capacity"])
            if side_metrics is not None
            else None
        ),
        "label_layout": "collision_aware",
    }

    metadata_path = output_path.with_name(
        f"{output_path.stem}.render.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path

def render_problem(
    problem: ProblemInstance,
    output_path: str | Path,
    cfg: RenderConfig,
    *,
    demand_encoding: DemandEncodingConfig | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_encoding = demand_encoding or DemandEncodingConfig()
    fig, ax, header_ax, side_ax, style = _base_axes(problem, cfg, active_encoding)
    _draw_header(header_ax, problem, active_encoding, style)
    _draw_nodes(ax, problem, style, active_encoding)
    _draw_side_panel(side_ax, problem, active_encoding, style, cfg)
    _save_figure(fig, output_path, cfg, pad_inches=0.08)
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
    routes_tuple = tuple(tuple(route) for route in routes)
    _validate_renderable_nodes(problem, routes_tuple)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_encoding = demand_encoding or DemandEncodingConfig()
    active_route_rendering = route_rendering or RouteRenderingConfig()
    fig, ax, header_ax, side_ax, style = _base_axes(problem, cfg, active_encoding)
    _draw_header(header_ax, problem, active_encoding, style)
    _draw_nodes(ax, problem, style, active_encoding)
    _draw_routes(ax, problem, routes_tuple, style, active_route_rendering)
    _draw_nodes(ax, problem, style, active_encoding)
    _draw_side_panel(side_ax, problem, active_encoding, style, cfg)
    _save_figure(fig, output_path, cfg, pad_inches=0.08)
    plt.close(fig)
    return output_path

def render_diagnostic_routes(problem: ProblemInstance, routes: Sequence[Iterable[int]], output_path: str | Path, cfg: RenderConfig, *, demand_encoding: DemandEncodingConfig | None = None, route_rendering: RouteRenderingConfig | None = None) -> Path:
    from .evaluation import validate_cvrp_routes
    routes_tuple = tuple(tuple(route) for route in routes)
    _validate_renderable_nodes(problem, routes_tuple)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_encoding = demand_encoding or DemandEncodingConfig()
    active_route_rendering = route_rendering or RouteRenderingConfig()
    fig, ax, header_ax, side_ax, style = _base_axes(problem, cfg, active_encoding)
    _draw_header(header_ax, problem, active_encoding, style)
    _draw_nodes(ax, problem, style, active_encoding)
    _draw_routes(ax, problem, routes_tuple, style, active_route_rendering)
    _draw_nodes(ax, problem, style, active_encoding)
    _draw_side_panel(side_ax, problem, active_encoding, style, cfg)
    validation = validate_cvrp_routes(problem, routes_tuple)
    route_load_text = ", ".join(f"R{index + 1}: {load}/{problem.capacity}" for index, load in enumerate(validation.route_loads))
    vehicle_limit = "unlimited" if problem.max_vehicles is None else str(problem.max_vehicles)
    status = "valid" if validation.valid else ", ".join(validation.reasons)
    fig.text(0.5, 0.015, f"Route loads: {route_load_text}   |   Vehicles: {validation.vehicle_count}/{vehicle_limit}   |   Validation: {status}", ha="center", va="bottom", fontsize=max(7.0, style.font_size - 1.0), color="#111827")
    _save_figure(fig, output_path, cfg, pad_inches=0.10)
    plt.close(fig)
    return output_path
