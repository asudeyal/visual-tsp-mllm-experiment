"""CVRP görsel üretiminin testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.instances import build_capacity_demo_10
from src.rendering import (
    DemandEncoding,
    render_problem,
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