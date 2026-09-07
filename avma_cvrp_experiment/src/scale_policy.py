# Versioned instance workspace-scale policies for AVMA-CVRP.

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import RenderConfig


DEFAULT_SCALE_POLICY_PATH = (
    "data/cvrplib/scale_policies/benchmark_scale_v1.json"
)
DEFAULT_SCALE_POLICY_NAME = "benchmark_scale_v1"

_PIXEL_FIELDS = (
    "canvas_width_px",
    "canvas_height_px",
    "map_width_px",
    "panel_width_px",
    "panel_header_height_px",
)


@dataclass(frozen=True)
class ScalePolicy:
    name: str
    path: str
    sha256: str
    instances: dict[str, float]

    def scale_for_instance(self, instance_name: str) -> float:
        key = Path(instance_name).name
        if key not in self.instances:
            raise ValueError(
                f"Frozen scale policy {self.name!r} does not define instance {key!r}. "
                "Add the instance to a versioned scale-policy file before running it."
            )
        return self.instances[key]


def load_scale_policy(path: str | Path) -> ScalePolicy:
    path = Path(path)
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        loaded = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid scale-policy JSON: {path}") from exc

    if not isinstance(loaded, dict):
        raise ValueError("Scale-policy JSON root must be an object")

    name = loaded.get("policy_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("scale policy requires a non-empty policy_name")

    raw_instances = loaded.get("instances")
    if not isinstance(raw_instances, dict) or not raw_instances:
        raise ValueError("scale policy requires a non-empty instances object")

    instances: dict[str, float] = {}
    for raw_key, raw_value in raw_instances.items():
        key = Path(str(raw_key)).name
        if not key:
            raise ValueError("scale-policy instance key may not be empty")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Scale for {key!r} must be numeric") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Scale for {key!r} must be finite and > 0; got {raw_value!r}"
            )
        instances[key] = value

    return ScalePolicy(
        name=name.strip(),
        path=str(path.resolve()),
        sha256=sha256,
        instances=instances,
    )


def scale_render_config(
    render: RenderConfig,
    workspace_scale: float,
) -> RenderConfig:
    scale = float(workspace_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("workspace_scale must be finite and > 0")

    field_names = set(getattr(render, "__dataclass_fields__", {}))
    changes: dict[str, Any] = {}

    if bool(getattr(render, "fixed_canvas", False)):
        for field in _PIXEL_FIELDS:
            if field not in field_names:
                continue
            value = getattr(render, field)
            if value is None:
                continue
            changes[field] = max(1, int(round(float(value) * scale)))
    elif "figure_size_inches" in field_names:
        changes["figure_size_inches"] = float(render.figure_size_inches) * scale

    return replace(render, **changes)


def scale_manifest_context(
    *,
    policy: ScalePolicy,
    instance_name: str,
    workspace_scale: float,
    render: RenderConfig,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    policy_path = Path(policy.path)

    try:
        displayed_path = str(policy_path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        displayed_path = str(policy_path)

    context: dict[str, Any] = {
        "scale_policy_name": policy.name,
        "scale_policy_path": displayed_path,
        "scale_policy_sha256": policy.sha256,
        "scale_instance_key": Path(instance_name).name,
        "workspace_scale": float(workspace_scale),
    }

    for field in _PIXEL_FIELDS:
        if hasattr(render, field):
            value = getattr(render, field)
            if value is not None:
                context[field] = int(value)

    return context
