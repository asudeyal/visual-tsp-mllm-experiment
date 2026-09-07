from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.experiment.manifest import assert_shared_manifest_compatible
from src.scale_policy import load_scale_policy, scale_render_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "cvrplib"
    / "scale_policies"
    / "benchmark_scale_v1.json"
)


def test_main_scale_policy_values() -> None:
    policy = load_scale_policy(POLICY_PATH)

    assert policy.name == "benchmark_scale_v1"
    assert policy.scale_for_instance("P-n21-k2.vrp") == pytest.approx(1.0)
    assert policy.scale_for_instance("A-n37-k5.vrp") == pytest.approx(1.0)
    assert policy.scale_for_instance("E-n51-k5.vrp") == pytest.approx(1.0)
    assert policy.scale_for_instance("X-n110-k13.vrp") == pytest.approx(2.092)
    assert policy.scale_for_instance("X-n204-k19.vrp") == pytest.approx(3.0)
    assert policy.scale_for_instance("X-n298-k31.vrp") == pytest.approx(3.0)
    assert policy.scale_for_instance("X-n393-k38.vrp") == pytest.approx(3.0)


def test_missing_instance_is_rejected() -> None:
    policy = load_scale_policy(POLICY_PATH)
    with pytest.raises(ValueError, match="does not define instance"):
        policy.scale_for_instance("UNKNOWN-n999.vrp")


def test_fixed_canvas_scaling_matches_frozen_qa_rule() -> None:
    cfg = load_config(
        PROJECT_ROOT
        / "configs"
        / "main_8method"
        / "bar_sidepanel.yaml"
    )
    scaled = scale_render_config(cfg.render, 2.092)

    for field in (
        "canvas_width_px",
        "canvas_height_px",
        "map_width_px",
        "panel_width_px",
        "panel_header_height_px",
    ):
        if hasattr(cfg.render, field):
            base = getattr(cfg.render, field)
            actual = getattr(scaled, field)
            if base is not None:
                assert actual == max(1, int(round(float(base) * 2.092)))


def test_manifest_rejects_scale_policy_change() -> None:
    keys = (
        "architecture_version",
        "information_policy",
        "git_commit_sha",
        "config_sha256",
        "prompt_set",
        "prompt_sha256",
        "problem",
        "run_parameters",
        "render_policy",
        "provider_policy",
    )

    existing = {key: None for key in keys}
    expected = {key: None for key in keys}

    existing["render_policy"] = {
        "scale_policy_name": "benchmark_scale_v1",
        "workspace_scale": 1.0,
    }
    expected["render_policy"] = {
        "scale_policy_name": "benchmark_scale_v2",
        "workspace_scale": 2.0,
    }

    with pytest.raises(ValueError, match="render_policy"):
        assert_shared_manifest_compatible(existing, expected)
