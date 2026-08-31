"""Configuration loading and validation for AVMA-CVRP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEMAND_ENCODING_MODES = frozenset(
    {"none", "size", "bar", "dot_density", "color"}
)

DEFAULT_COLOR_STOPS = (
    "#FFFFFF",
    "#C6DBEF",
    "#6BAED6",
    "#DE2D26",
    "#111111",
)

DEFAULT_ROUTE_PALETTE = (
    "#E15759",
    "#4E79A7",
    "#59A14F",
    "#B07AA1",
    "#F28E2B",
    "#76B7B2",
)


@dataclass(frozen=True)
class AgentConfig:
    temperature: float | None = None
    thinking_level: str | None = None
    max_output_retries: int = 2
    candidates: int = 1
    max_attempts: int = 1


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    candidate_strategy: str = "independent_calls"
    timeout_seconds: int = 120
    request_retries: int = 3


@dataclass(frozen=True)
class StagnationConfig:
    window: int = 5
    similarity_threshold: float = 0.90
    max_unique_routes: int = 2


@dataclass(frozen=True)
class RenderConfig:
    figure_size_inches: float = 8.0
    dpi: int = 150
    padding_ratio: float = 0.08
    node_size: int = 440
    font_size: int = 9
    route_line_width: float = 1.8


@dataclass(frozen=True)
class DemandEncodingConfig:
    """Visual-only customer-demand representation settings.

    These values control rendering only. Numeric demand and capacity values
    remain hidden from model prompts and are never rendered as text.
    """

    mode: str = "none"
    show_visual_legend: bool = True
    show_vehicle_icons: bool = True
    size_min_factor: float = 0.42
    size_max_factor: float = 1.00
    bar_width_points: float = 30.0
    bar_height_points: float = 5.0
    dot_grid_size: int = 7
    dot_radius_points: float = 1.2
    color_stops: tuple[str, ...] = DEFAULT_COLOR_STOPS


@dataclass(frozen=True)
class RouteRenderingConfig:
    """Visual style for vehicle-route edges."""

    palette: tuple[str, ...] = DEFAULT_ROUTE_PALETTE


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_label: str
    iterations: int
    seed: int
    prompt_set: str
    output_dir: str
    strict_euc_2d: bool
    max_restart_attempts: int
    provider: ProviderConfig
    initializer: AgentConfig
    critic: AgentConfig
    scorer: AgentConfig
    repair: AgentConfig
    hybrid: AgentConfig
    diversity: AgentConfig
    stagnation: StagnationConfig
    render: RenderConfig
    demand_encoding: DemandEncodingConfig
    route_rendering: RouteRenderingConfig
    raw: dict[str, Any]
    source_path: str
    sha256: str


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} YAML içinde bir nesne olmalıdır")
    return dict(value)


def _string_tuple(
    value: Any,
    *,
    field_name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} bir metin listesi olmalıdır")

    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} yalnız boş olmayan metinler içermelidir")
        parsed.append(item.strip())

    return tuple(parsed)


def _agent(raw: dict[str, Any], name: str, **defaults: Any) -> AgentConfig:
    agents_raw = _mapping(raw.get("agents"), "agents")
    section = dict(defaults)
    section.update(_mapping(agents_raw.get(name), f"agents.{name}"))

    return AgentConfig(
        temperature=section.get("temperature"),
        thinking_level=section.get("thinking_level"),
        max_output_retries=int(section.get("max_output_retries", 2)),
        candidates=int(section.get("candidates", 1)),
        max_attempts=int(section.get("max_attempts", 1)),
    )


def load_config(
    path: str | Path,
    *,
    provider_name: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)

    if loaded is None:
        raw: dict[str, Any] = {}
    elif isinstance(loaded, dict):
        raw = dict(loaded)
    else:
        raise ValueError("Config YAML kökü bir nesne olmalıdır")

    exp = _mapping(raw.get("experiment"), "experiment")
    provider_raw = _mapping(raw.get("provider"), "provider")
    stagnation_raw = _mapping(raw.get("stagnation"), "stagnation")
    render_raw = _mapping(raw.get("render"), "render")
    demand_encoding_raw = _mapping(
        raw.get("demand_encoding"),
        "demand_encoding",
    )
    route_rendering_raw = _mapping(
        raw.get("route_rendering"),
        "route_rendering",
    )

    name = str(exp.get("name", path.stem))
    iterations = int(exp.get("iterations", 20))

    config = ExperimentConfig(
        name=name,
        run_label=str(exp.get("run_label", f"i{iterations}")),
        iterations=iterations,
        seed=int(seed if seed is not None else exp.get("seed", 42)),
        prompt_set=str(exp.get("prompt_set", "v1")),
        output_dir=str(exp.get("output_dir", "output/runs")),
        strict_euc_2d=bool(exp.get("strict_euc_2d", True)),
        max_restart_attempts=int(exp.get("max_restart_attempts", 3)),
        provider=ProviderConfig(
            name=str(provider_name or provider_raw.get("name") or "").strip(),
            model=str(model or provider_raw.get("model") or "").strip(),
            candidate_strategy=str(
                provider_raw.get("candidate_strategy", "independent_calls")
            ),
            timeout_seconds=int(provider_raw.get("timeout_seconds", 120)),
            request_retries=int(provider_raw.get("request_retries", 3)),
        ),
        initializer=_agent(raw, "initializer", max_attempts=1),
        critic=_agent(raw, "critic", candidates=3),
        scorer=_agent(raw, "scorer"),
        repair=_agent(raw, "repair", max_attempts=2),
        hybrid=_agent(raw, "hybrid"),
        diversity=_agent(raw, "diversity"),
        stagnation=StagnationConfig(
            window=int(stagnation_raw.get("window", 5)),
            similarity_threshold=float(
                stagnation_raw.get("similarity_threshold", 0.90)
            ),
            max_unique_routes=int(
                stagnation_raw.get("max_unique_routes", 2)
            ),
        ),
        render=RenderConfig(
            figure_size_inches=float(
                render_raw.get("figure_size_inches", 8.0)
            ),
            dpi=int(render_raw.get("dpi", 150)),
            padding_ratio=float(render_raw.get("padding_ratio", 0.08)),
            node_size=int(render_raw.get("node_size", 440)),
            font_size=int(render_raw.get("font_size", 9)),
            route_line_width=float(
                render_raw.get("route_line_width", 1.8)
            ),
        ),
        demand_encoding=DemandEncodingConfig(
            mode=str(demand_encoding_raw.get("mode", "none")).strip().lower(),
            show_visual_legend=bool(
                demand_encoding_raw.get("show_visual_legend", True)
            ),
            show_vehicle_icons=bool(
                demand_encoding_raw.get("show_vehicle_icons", True)
            ),
            size_min_factor=float(
                demand_encoding_raw.get("size_min_factor", 0.42)
            ),
            size_max_factor=float(
                demand_encoding_raw.get("size_max_factor", 1.00)
            ),
            bar_width_points=float(
                demand_encoding_raw.get("bar_width_points", 30.0)
            ),
            bar_height_points=float(
                demand_encoding_raw.get("bar_height_points", 5.0)
            ),
            dot_grid_size=int(
                demand_encoding_raw.get("dot_grid_size", 7)
            ),
            dot_radius_points=float(
                demand_encoding_raw.get("dot_radius_points", 1.2)
            ),
            color_stops=_string_tuple(
                demand_encoding_raw.get("color_stops"),
                field_name="demand_encoding.color_stops",
                default=DEFAULT_COLOR_STOPS,
            ),
        ),
        route_rendering=RouteRenderingConfig(
            palette=_string_tuple(
                route_rendering_raw.get("palette"),
                field_name="route_rendering.palette",
                default=DEFAULT_ROUTE_PALETTE,
            ),
        ),
        raw=raw,
        source_path=str(path),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )

    validate_config(config)
    return config


def _validate_agent(name: str, config: AgentConfig) -> None:
    if config.max_output_retries < 0:
        raise ValueError(
            f"agents.{name}.max_output_retries negatif olamaz"
        )
    if config.candidates < 1:
        raise ValueError(f"agents.{name}.candidates en az 1 olmalıdır")
    if config.max_attempts < 1:
        raise ValueError(f"agents.{name}.max_attempts en az 1 olmalıdır")


def validate_config(config: ExperimentConfig) -> None:
    if not config.provider.name:
        raise ValueError("provider CLI ile veya legacy config içinde verilmelidir")

    if not config.provider.model:
        raise ValueError("model CLI ile veya legacy config içinde verilmelidir")

    if not config.run_label.strip():
        raise ValueError("experiment.run_label boş olamaz")

    if not config.prompt_set.strip():
        raise ValueError("experiment.prompt_set boş olamaz")

    if config.iterations < 1:
        raise ValueError("iterations en az 1 olmalıdır")

    if config.max_restart_attempts < 1:
        raise ValueError("max_restart_attempts en az 1 olmalıdır")

    if config.provider.timeout_seconds < 1:
        raise ValueError("provider.timeout_seconds en az 1 olmalıdır")

    if config.provider.request_retries < 1:
        raise ValueError("provider.request_retries en az 1 olmalıdır")

    if config.provider.candidate_strategy != "independent_calls":
        raise ValueError(
            "AVMA karşılaştırmalarında "
            "Critic candidate_strategy='independent_calls' olmalıdır"
        )

    _validate_agent("initializer", config.initializer)
    _validate_agent("critic", config.critic)
    _validate_agent("scorer", config.scorer)
    _validate_agent("repair", config.repair)
    _validate_agent("hybrid", config.hybrid)
    _validate_agent("diversity", config.diversity)

    if config.stagnation.window < 2:
        raise ValueError("stagnation.window en az 2 olmalıdır")

    if not 0.0 <= config.stagnation.similarity_threshold <= 1.0:
        raise ValueError(
            "stagnation.similarity_threshold 0 ile 1 arasında olmalıdır"
        )

    if config.stagnation.max_unique_routes < 1:
        raise ValueError(
            "stagnation.max_unique_routes en az 1 olmalıdır"
        )

    if config.render.figure_size_inches <= 0:
        raise ValueError("render.figure_size_inches pozitif olmalıdır")

    if config.render.dpi < 1:
        raise ValueError("render.dpi en az 1 olmalıdır")

    if not 0.0 <= config.render.padding_ratio < 1.0:
        raise ValueError("render.padding_ratio 0 ile 1 arasında olmalıdır")

    if config.render.node_size < 1:
        raise ValueError("render.node_size en az 1 olmalıdır")

    if config.render.font_size <= 0:
        raise ValueError("render.font_size pozitif olmalıdır")

    if config.render.route_line_width <= 0:
        raise ValueError("render.route_line_width pozitif olmalıdır")

    encoding = config.demand_encoding

    if encoding.mode not in DEMAND_ENCODING_MODES:
        allowed = ", ".join(sorted(DEMAND_ENCODING_MODES))
        raise ValueError(
            f"demand_encoding.mode bilinmiyor: {encoding.mode}. "
            f"İzin verilenler: {allowed}"
        )

    if encoding.size_min_factor <= 0:
        raise ValueError(
            "demand_encoding.size_min_factor pozitif olmalıdır"
        )

    if encoding.size_max_factor <= 0:
        raise ValueError(
            "demand_encoding.size_max_factor pozitif olmalıdır"
        )

    if encoding.size_min_factor > encoding.size_max_factor:
        raise ValueError(
            "demand_encoding.size_min_factor, "
            "size_max_factor değerinden büyük olamaz"
        )

    if encoding.bar_width_points <= 0:
        raise ValueError(
            "demand_encoding.bar_width_points pozitif olmalıdır"
        )

    if encoding.bar_height_points <= 0:
        raise ValueError(
            "demand_encoding.bar_height_points pozitif olmalıdır"
        )

    if encoding.dot_grid_size < 2:
        raise ValueError(
            "demand_encoding.dot_grid_size en az 2 olmalıdır"
        )

    if encoding.dot_radius_points <= 0:
        raise ValueError(
            "demand_encoding.dot_radius_points pozitif olmalıdır"
        )

    if len(encoding.color_stops) < 2:
        raise ValueError(
            "demand_encoding.color_stops en az iki renk içermelidir"
        )

    if not config.route_rendering.palette:
        raise ValueError(
            "route_rendering.palette en az bir rota rengi içermelidir"
        )


def config_as_json(config: ExperimentConfig) -> str:
    return json.dumps(
        config.raw,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )