from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config


def _write_config(path: Path, *, layout: str, media_resolution: str = "high") -> None:
    path.write_text(
        f"""
experiment:
  name: test
  run_label: test
  iterations: 1
  prompt_set: cvrp_capacity_v2
provider:
  name: gemini
  model: gemini-3.8-flash
  candidate_strategy: independent_calls
  media_resolution: {media_resolution}
demand_encoding:
  mode: bar
  bar_layout: {layout}
""".strip() + "\n",
        encoding="utf-8",
    )


def test_bar_layout_is_parsed(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path, layout="collision_aware")
    config = load_config(path)
    assert config.demand_encoding.bar_layout == "collision_aware"
    assert config.provider.media_resolution == "high"


def test_unknown_bar_layout_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path, layout="unknown")
    with pytest.raises(ValueError, match="bar_layout"):
        load_config(path)


def test_unknown_media_resolution_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path, layout="local", media_resolution="ultra_high")
    with pytest.raises(ValueError, match="media_resolution"):
        load_config(path)
