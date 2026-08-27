"""Shared data models for AVMA-TSP.

The most important design rule is the information firewall: model-facing objects and
observer/evaluation objects are intentionally separate. Agent code should never need
an ObserverEvaluation to build a prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ProblemInstance:
    name: str
    dimension: int
    node_ids: tuple[int, ...]
    coordinates: dict[int, tuple[float, float]]
    depot: int
    edge_weight_type: str = "EUC_2D"
    source_path: str | None = None
    source_sha256: str | None = None
    reference_optimum: float | None = None

    def to_public_metadata(self) -> dict[str, Any]:
        """Metadata safe for manifests, never for model prompts."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "depot": self.depot,
            "edge_weight_type": self.edge_weight_type,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "reference_optimum": self.reference_optimum,
        }


@dataclass(frozen=True)
class PromptPart:
    kind: Literal["text", "image"]
    value: str | Path
    label: str | None = None


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_vision: bool = True
    supports_multiple_images: bool = True
    supports_native_multiple_choices: bool = False
    supports_temperature: bool = True
    supports_thinking_level: bool = False
    max_native_choices: int = 1


@dataclass
class ProviderResponse:
    text: str
    provider: str
    model: str
    phase: str
    latency_seconds: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderManyResponse:
    responses: list[ProviderResponse]
    strategy: str


@dataclass(frozen=True)
class RouteCandidate:
    candidate_id: int
    route: tuple[int, ...]
    source: str
    raw_text: str
    image_path: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    renderable: bool
    missing_nodes: tuple[int, ...]
    duplicate_nodes: tuple[int, ...]
    unknown_nodes: tuple[int, ...]
    starts_at_depot: bool
    ends_at_depot: bool
    closed_cycle: bool
    expected_route_length: int
    observed_route_length: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObserverEvaluation:
    validation: ValidationResult
    distance: float | None
    gap_percent: float | None
    crossings: int | None
    canonical_route: tuple[int, ...] | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.canonical_route is not None:
            data["canonical_route"] = list(self.canonical_route)
        return data


@dataclass(frozen=True)
class StructuralStagnationResult:
    stagnated: bool
    window_size: int
    unique_canonical_routes: int
    exact_repeat_signal: bool
    mean_consecutive_similarity: float
    similarity_signal: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentCallRecord:
    agent: str
    prompt_text: str
    request_parts: list[dict[str, str]]
    raw_response: str
    provider_response: ProviderResponse
    failed_output_attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "prompt_text": self.prompt_text,
            "request_parts": self.request_parts,
            "raw_response": self.raw_response,
            "provider": self.provider_response.provider,
            "model": self.provider_response.model,
            "phase": self.provider_response.phase,
            "latency_seconds": self.provider_response.latency_seconds,
            "usage": self.provider_response.usage,
            "raw_metadata": self.provider_response.raw_metadata,
            "failed_output_attempts": self.failed_output_attempts,
        }


@dataclass
class RouteAgentResult:
    candidate: RouteCandidate
    call: AgentCallRecord


@dataclass
class CriticResult:
    candidates: list[RouteCandidate]
    calls: list[AgentCallRecord]


@dataclass
class ScorerResult:
    ranking: list[int]
    best_id: int
    call: AgentCallRecord


@dataclass
class HybridResult:
    candidate: RouteCandidate
    selected_edges: tuple[tuple[int, int], tuple[int, int]]
    call: AgentCallRecord


@dataclass
class CheckpointState:
    completed_iteration: int
    working_route: list[int]
    structural_history: list[list[int]]
    hybrid_used_since_restart: bool
    restart_count: int
    observed_oracle_best_distance: float | None
    observed_oracle_best_route: list[int] | None
    selected_best_distance: float | None
    selected_best_route: list[int] | None
    config_sha256: str
    instance_sha256: str | None
