"""CVRP görsel üretiminin testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.instances import build_capacity_demo_10
from src.rendering import (
    DemandEncoding,
    _customer_marker_colors,
    _customer_marker_areas,
    render_problem,
    render_solution,
)


def test_numeric_render_creates_valid_png(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / "numeric.png"

    result_path = render_problem(
        problem,
        output_path,
        encoding=DemandEncoding.NUMERIC,
    )

    assert result_path == output_path
    assert output_path.is_file()

    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size == (1600, 1200)
        assert image.mode in {
            "RGB",
            "RGBA",
        }


def test_rendered_image_is_not_blank(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / "numeric.png"

    render_problem(
        problem,
        output_path,
    )

    with Image.open(output_path) as image:
        rgb_image = image.convert("RGB")
        extrema = rgb_image.getextrema()

    assert any(
        minimum != maximum
        for minimum, maximum in extrema
    )


def test_size_render_creates_valid_png(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / "size.png"

    render_problem(
        problem,
        output_path,
        encoding=DemandEncoding.SIZE,
    )

    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size == (1600, 1200)


def test_size_marker_area_is_proportional_to_demand(
) -> None:
    problem = build_capacity_demo_10()

    size_areas = _customer_marker_areas(
        problem,
        encoding=DemandEncoding.SIZE,
    )
    numeric_areas = _customer_marker_areas(
        problem,
        encoding=DemandEncoding.NUMERIC,
    )

    demand_to_area = {
        customer.demand: area
        for customer, area in zip(
            problem.customers,
            size_areas,
            strict=True,
        )
    }

    assert demand_to_area[2] == pytest.approx(
        demand_to_area[1] * 2
    )
    assert demand_to_area[3] == pytest.approx(
        demand_to_area[1] * 3
    )
    assert len(set(numeric_areas)) == 1


def test_color_render_creates_valid_png(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / "color.png"

    render_problem(
        problem,
        output_path,
        encoding=DemandEncoding.COLOR,
    )

    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size == (1600, 1200)


def test_color_encoding_uses_one_color_per_demand(
) -> None:
    problem = build_capacity_demo_10()

    colors = _customer_marker_colors(
        problem,
        encoding=DemandEncoding.COLOR,
    )
    areas = _customer_marker_areas(
        problem,
        encoding=DemandEncoding.COLOR,
    )
    demand_to_color = {
        customer.demand: color
        for customer, color in zip(
            problem.customers,
            colors,
            strict=True,
        )
    }

    assert len(demand_to_color) == 3
    assert len(set(demand_to_color.values())) == 3
    assert len(set(areas)) == 1


def test_render_creates_parent_directories(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = (
        tmp_path
        / "nested"
        / "images"
        / "numeric.png"
    )

    render_problem(
        problem,
        output_path,
        encoding="numeric",
    )

    assert output_path.is_file()


def test_unknown_encoding_is_rejected(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()

    with pytest.raises(
        ValueError,
        match="Desteklenmeyen talep gösterimi",
    ):
        render_problem(
            problem,
            tmp_path / "invalid.png",
            encoding="unknown",
        )


def test_non_png_output_is_rejected(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()

    with pytest.raises(
        ValueError,
        match=r"\.png",
    ):
        render_problem(
            problem,
            tmp_path / "problem.jpg",
        )


def test_solution_render_creates_valid_png(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / "solution.png"

    result_path = render_solution(
        problem,
        (
            (0, 9, 2, 1, 0),
            (0, 8, 5, 3, 0),
            (0, 7, 6, 4, 0),
        ),
        output_path,
        title="Exact CVRP baseline",
        route_loads=(6, 6, 6),
    )

    assert result_path == output_path

    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size == (1600, 1200)
        assert image.mode in {
            "RGB",
            "RGBA",
        }


def test_solution_render_calculates_loads(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / "solution.png"

    render_solution(
        problem,
        (
            (0, 9, 2, 1, 0),
            (0, 8, 5, 3, 0),
            (0, 7, 6, 4, 0),
        ),
        output_path,
        title="Calculated loads",
    )

    assert output_path.is_file()


def test_solution_render_rejects_unknown_node(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()

    with pytest.raises(
        ValueError,
        match="bilinmeyen düğümler",
    ):
        render_solution(
            problem,
            ((0, 99, 0),),
            tmp_path / "invalid.png",
            title="Invalid",
        )


def test_solution_render_rejects_load_count_mismatch(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()

    with pytest.raises(
        ValueError,
        match="Rota yüklerinin sayısı",
    ):
        render_solution(
            problem,
            (
                (0, 1, 0),
                (0, 2, 0),
            ),
            tmp_path / "invalid.png",
            title="Invalid",
            route_loads=(2,),
        )
