"""CVRP görsel üretiminin testleri."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from matplotlib.colors import to_rgb
from PIL import Image

from src.instances import build_capacity_demo_10
from src.rendering import (
    DemandEncoding,
    _capacity_bar_fill_fractions,
    _customer_marker_colors,
    _customer_marker_diameters,
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


def test_size_diameter_is_normalized_by_capacity(
) -> None:
    problem = build_capacity_demo_10()

    diameters = _customer_marker_diameters(
        problem,
        encoding=DemandEncoding.SIZE,
    )
    larger_capacity_problem = replace(
        problem,
        vehicle_capacity=12,
    )
    larger_capacity_diameters = (
        _customer_marker_diameters(
            larger_capacity_problem,
            encoding=DemandEncoding.SIZE,
        )
    )
    numeric_areas = _customer_marker_areas(
        problem,
        encoding=DemandEncoding.NUMERIC,
    )

    demand_to_diameter = {
        customer.demand: diameter
        for customer, diameter in zip(
            problem.customers,
            diameters,
            strict=True,
        )
    }
    larger_capacity_by_demand = {
        customer.demand: diameter
        for customer, diameter in zip(
            larger_capacity_problem.customers,
            larger_capacity_diameters,
            strict=True,
        )
    }

    first_step = (
        demand_to_diameter[2]
        - demand_to_diameter[1]
    )
    second_step = (
        demand_to_diameter[3]
        - demand_to_diameter[2]
    )
    larger_capacity_span = (
        larger_capacity_by_demand[3]
        - larger_capacity_by_demand[1]
    )
    original_span = (
        demand_to_diameter[3]
        - demand_to_diameter[1]
    )

    assert first_step == pytest.approx(second_step)
    assert larger_capacity_span == pytest.approx(
        original_span / 2
    )
    assert len(set(numeric_areas)) == 1


def test_color_intensity_render_creates_valid_png(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / "color_intensity.png"

    render_problem(
        problem,
        output_path,
        encoding=DemandEncoding.COLOR_INTENSITY,
    )

    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size == (1600, 1200)


def test_color_intensity_is_normalized_by_capacity(
) -> None:
    problem = build_capacity_demo_10()

    colors = _customer_marker_colors(
        problem,
        encoding=DemandEncoding.COLOR_INTENSITY,
    )
    demand_to_color = {
        customer.demand: color
        for customer, color in zip(
            problem.customers,
            colors,
            strict=True,
        )
    }
    brightness_by_demand = {
        demand: sum(to_rgb(color))
        for demand, color in demand_to_color.items()
    }
    larger_capacity_problem = replace(
        problem,
        vehicle_capacity=12,
    )
    larger_capacity_colors = _customer_marker_colors(
        larger_capacity_problem,
        encoding=DemandEncoding.COLOR_INTENSITY,
    )
    larger_capacity_by_demand = {
        customer.demand: color
        for customer, color in zip(
            larger_capacity_problem.customers,
            larger_capacity_colors,
            strict=True,
        )
    }

    assert brightness_by_demand[1] > (
        brightness_by_demand[2]
    )
    assert brightness_by_demand[2] > (
        brightness_by_demand[3]
    )
    assert sum(
        to_rgb(larger_capacity_by_demand[3])
    ) > sum(to_rgb(demand_to_color[3]))


def test_bar_length_render_creates_valid_png(
    tmp_path: Path,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / "bar_length.png"

    render_problem(
        problem,
        output_path,
        encoding=DemandEncoding.BAR_LENGTH,
    )

    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size == (1600, 1200)


def test_bar_length_uses_demand_capacity_ratio(
) -> None:
    problem = build_capacity_demo_10()

    fill_fractions = _capacity_bar_fill_fractions(
        problem
    )
    demand_to_fraction = {
        customer.demand: fill_fraction
        for customer, fill_fraction in zip(
            problem.customers,
            fill_fractions,
            strict=True,
        )
    }
    areas = _customer_marker_areas(
        problem,
        encoding=DemandEncoding.BAR_LENGTH,
    )

    assert demand_to_fraction == pytest.approx(
        {
            1: 1 / 6,
            2: 2 / 6,
            3: 3 / 6,
        }
    )
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
