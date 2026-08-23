"""Yeni görsel talep kodlamalarının testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from run_refinement import SUPPORTED_ENCODINGS
from src.instances import build_capacity_demo_10
from src.model_contract import build_solver_prompt
from src.rendering import (
    DemandEncoding,
    _customer_marker_areas,
    render_problem,
)


@pytest.mark.parametrize(
    "encoding",
    [
        DemandEncoding.SCALE_POSITION,
        DemandEncoding.RADIAL_FILL,
    ],
)
def test_new_encoding_render_creates_valid_png(
    tmp_path: Path,
    encoding: DemandEncoding,
) -> None:
    problem = build_capacity_demo_10()
    output_path = tmp_path / f"{encoding.value}.png"

    render_problem(
        problem,
        output_path,
        encoding=encoding,
    )

    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.size == (1600, 1200)

    areas = _customer_marker_areas(
        problem,
        encoding=encoding,
    )
    assert len(set(areas)) == 1


def test_scale_position_prompt_describes_fixed_scale() -> None:
    prompt = build_solver_prompt(
        encoding=DemandEncoding.SCALE_POSITION,
    )

    assert "vertical position" in prompt
    assert "triangular pointer" in prompt
    assert "bottom endpoint represents zero demand" in prompt
    assert "top endpoint represents vehicle capacity Q" in prompt
    assert "same length" in prompt
    assert "demand divided by Q" in prompt
    assert "d=<value>" not in prompt


def test_radial_fill_prompt_describes_clockwise_arc() -> None:
    prompt = build_solver_prompt(
        encoding=DemandEncoding.RADIAL_FILL,
    )

    assert "amber filled arc" in prompt
    assert "fixed-size ring" in prompt
    assert "empty gray ring" in prompt
    assert "starts at 12 o'clock" in prompt
    assert "increases clockwise" in prompt
    assert "demand divided by Q" in prompt
    assert "d=<value>" not in prompt


def test_refinement_supports_new_encodings() -> None:
    assert DemandEncoding.SCALE_POSITION in SUPPORTED_ENCODINGS
    assert DemandEncoding.RADIAL_FILL in SUPPORTED_ENCODINGS
