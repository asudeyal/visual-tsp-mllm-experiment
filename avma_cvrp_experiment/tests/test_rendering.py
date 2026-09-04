from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from src.config import DemandEncodingConfig, RenderConfig
from src.rendering import render_problem, render_routes
from src.schemas import ProblemInstance


def _problem() -> ProblemInstance:
    coordinates = {
        1: (0.0, 0.0),
        2: (1.0, 1.0),
        3: (1.3, 1.1),
        4: (1.6, 1.0),
        5: (3.0, 3.0),
        6: (3.2, 3.1),
        7: (5.0, 1.0),
        8: (5.2, 1.2),
    }
    return ProblemInstance(
        name="synthetic-cvrp",
        dimension=len(coordinates),
        node_ids=tuple(coordinates),
        coordinates=coordinates,
        depot=1,
        capacity=100,
        demands={1: 0, 2: 20, 3: 40, 4: 60, 5: 80, 6: 30, 7: 50, 8: 70},
        max_vehicles=3,
    )


def _cfg() -> RenderConfig:
    return RenderConfig(
        figure_size_inches=4.0,
        dpi=100,
        padding_ratio=0.10,
        node_size=260,
        font_size=8,
        route_line_width=1.4,
        fixed_canvas=True,
    )


def _encoding(layout: str) -> DemandEncodingConfig:
    return DemandEncodingConfig(
        mode="bar",
        bar_layout=layout,
        show_visual_legend=True,
        show_vehicle_icons=True,
        bar_width_points=28,
        bar_height_points=5,
        side_panel_width_inches=2.0,
        side_panel_columns=2,
        side_bar_width_points=38,
        side_bar_height_points=7,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_bar_layouts_render(tmp_path: Path) -> None:
    problem = _problem()
    cfg = _cfg()
    for layout in ("local", "collision_aware", "side_panel"):
        path = tmp_path / f"{layout}.png"
        render_problem(problem, path, cfg, demand_encoding=_encoding(layout))
        assert path.exists()
        metadata = json.loads((tmp_path / f"{layout}.render.json").read_text(encoding="utf-8"))
        assert metadata["bar_layout"] == layout
        assert metadata["fixed_canvas"] is True
        assert metadata["sha256"] == _sha(path)


def test_fixed_canvas_is_deterministic(tmp_path: Path) -> None:
    problem = _problem()
    cfg = _cfg()
    encoding = _encoding("collision_aware")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    render_problem(problem, first, cfg, demand_encoding=encoding)
    render_problem(problem, second, cfg, demand_encoding=encoding)
    assert _sha(first) == _sha(second)


def test_side_panel_keeps_map_height_and_adds_width(tmp_path: Path) -> None:
    problem = _problem()
    cfg = _cfg()
    local = tmp_path / "local.png"
    side = tmp_path / "side.png"
    render_problem(problem, local, cfg, demand_encoding=_encoding("local"))
    render_problem(problem, side, cfg, demand_encoding=_encoding("side_panel"))
    with Image.open(local) as local_image, Image.open(side) as side_image:
        assert side_image.height == local_image.height
        assert side_image.width > local_image.width


def test_side_panel_route_image_matches_problem_dimensions(tmp_path: Path) -> None:
    problem = _problem()
    cfg = _cfg()
    encoding = _encoding("side_panel")
    problem_path = tmp_path / "problem.png"
    route_path = tmp_path / "routes.png"
    routes = ((1, 2, 3, 4, 1), (1, 5, 6, 1), (1, 7, 8, 1))
    render_problem(problem, problem_path, cfg, demand_encoding=encoding)
    render_routes(problem, routes, route_path, cfg, demand_encoding=encoding)
    with Image.open(problem_path) as problem_image, Image.open(route_path) as route_image:
        assert problem_image.size == route_image.size
