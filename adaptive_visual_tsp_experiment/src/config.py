"""Configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    raw: dict[str, Any]
    source_path: str
    sha256: str


def _agent(raw: dict[str, Any], name: str, **defaults: Any) -> AgentConfig:
    section = dict(defaults)
    section.update(raw.get("agents", {}).get(name, {}) or {})
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
    raw = yaml.safe_load(text) or {}

    exp = raw.get("experiment", {})
    provider_raw = raw.get("provider", {})
    stagnation_raw = raw.get("stagnation", {})
    render_raw = raw.get("render", {})

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
            candidate_strategy=str(provider_raw.get("candidate_strategy", "independent_calls")),
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
            similarity_threshold=float(stagnation_raw.get("similarity_threshold", 0.90)),
            max_unique_routes=int(stagnation_raw.get("max_unique_routes", 2)),
        ),
        render=RenderConfig(
            figure_size_inches=float(render_raw.get("figure_size_inches", 8.0)),
            dpi=int(render_raw.get("dpi", 150)),
            padding_ratio=float(render_raw.get("padding_ratio", 0.08)),
            node_size=int(render_raw.get("node_size", 440)),
            font_size=int(render_raw.get("font_size", 9)),
            route_line_width=float(render_raw.get("route_line_width", 1.8)),
        ),
        raw=raw,
        source_path=str(path),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if not config.provider.name:
        raise ValueError("provider CLI ile veya legacy config içinde verilmelidir")
    if not config.provider.model:
        raise ValueError("model CLI ile veya legacy config içinde verilmelidir")
    if not config.run_label.strip():
        raise ValueError("experiment.run_label boş olamaz")
    if config.iterations < 1:
        raise ValueError("iterations en az 1 olmalıdır")
    if config.critic.candidates < 1:
        raise ValueError("critic.candidates en az 1 olmalıdır")
    if config.repair.max_attempts < 1:
        raise ValueError("repair.max_attempts en az 1 olmalıdır")
    if config.stagnation.window < 2:
        raise ValueError("stagnation.window en az 2 olmalıdır")
    if not 0.0 <= config.stagnation.similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold 0 ile 1 arasında olmalıdır")
    if config.stagnation.max_unique_routes < 1:
        raise ValueError("max_unique_routes en az 1 olmalıdır")
    if config.max_restart_attempts < 1:
        raise ValueError("max_restart_attempts en az 1 olmalıdır")
    if config.provider.candidate_strategy != "independent_calls":
        raise ValueError(
            "AVMA karşılaştırmalarında Critic candidate_strategy='independent_calls' olmalıdır"
        )


def config_as_json(config: ExperimentConfig) -> str:
    return json.dumps(config.raw, ensure_ascii=False, indent=2, sort_keys=True)
