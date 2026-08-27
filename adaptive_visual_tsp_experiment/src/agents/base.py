"""Shared agent utilities and strict structured-output parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..config import AgentConfig
from ..prompts import PromptSet
from ..providers.base import ProviderAdapter
from ..schemas import AgentCallRecord, PromptPart, ProviderResponse, RouteCandidate


class ModelOutputError(ValueError):
    def __init__(self, message: str, *, attempts: Sequence[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts or ())


def output_attempt_record(
    response: ProviderResponse,
    error: Exception,
    *,
    attempt: int,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "phase": response.phase,
        "provider": response.provider,
        "model": response.model,
        "raw_response": response.text,
        "latency_seconds": response.latency_seconds,
        "usage": dict(response.usage),
        "raw_metadata": dict(response.raw_metadata),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ModelOutputError("JSON object bulunamadı")
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"Geçersiz JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelOutputError("JSON çıktı object olmalıdır")
    return value


def _strict_integer(value: Any, *, field: str) -> int:
    """Accept JSON integers or ASCII digit-only strings; reject all other coercions."""
    if isinstance(value, bool):
        raise ModelOutputError(f"{field} integer olmalıdır")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value and value.isascii() and value.isdigit():
        return int(value)
    raise ModelOutputError(f"{field} integer olmalıdır")


def parse_route(text: str) -> tuple[int, ...]:
    value = extract_json_object(text)
    route = value.get("route")
    if not isinstance(route, list) or len(route) < 2:
        raise ModelOutputError("route en az iki elemanlı liste olmalıdır")
    return tuple(_strict_integer(item, field="route node ID") for item in route)


def parse_scorer(text: str, expected_ids: set[int]) -> tuple[list[int], int]:
    value = extract_json_object(text)
    ranking_raw = value.get("ranking")
    if not isinstance(ranking_raw, list):
        raise ModelOutputError("ranking integer liste olmalıdır")
    ranking = [_strict_integer(item, field="ranking candidate ID") for item in ranking_raw]
    best_id = _strict_integer(value.get("best_id"), field="best_id")
    if set(ranking) != expected_ids or len(ranking) != len(expected_ids):
        raise ModelOutputError("ranking tüm candidate ID'lerini tam bir kez içermelidir")
    if best_id not in expected_ids or ranking[0] != best_id:
        raise ModelOutputError("best_id ranking'in ilk elemanı olmalıdır")
    return ranking, best_id


def parse_hybrid(text: str) -> tuple[tuple[int, ...], tuple[tuple[int, int], tuple[int, int]]]:
    value = extract_json_object(text)
    route_raw = value.get("route")
    edges_raw = value.get("selected_edges")
    if not isinstance(route_raw, list) or len(route_raw) < 2:
        raise ModelOutputError("Hybrid route geçersiz")
    route = tuple(_strict_integer(item, field="Hybrid route node ID") for item in route_raw)
    if (
        not isinstance(edges_raw, list)
        or len(edges_raw) != 2
        or any(not isinstance(edge, list) or len(edge) != 2 for edge in edges_raw)
    ):
        raise ModelOutputError("selected_edges tam olarak iki integer edge içermelidir")
    selected = tuple(
        tuple(_strict_integer(item, field="selected_edges node ID") for item in edge)
        for edge in edges_raw
    )
    return route, selected  # type: ignore[return-value]


def request_parts_manifest(parts: Sequence[PromptPart]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for part in parts:
        if part.kind == "text":
            result.append({"kind": "text", "label": part.label or "", "value": str(part.value)})
        else:
            result.append({"kind": "image", "label": part.label or "", "value": str(part.value)})
    return result


class BaseAgent:
    role_name: str

    def __init__(
        self,
        provider: ProviderAdapter,
        prompts: PromptSet,
        config: AgentConfig,
    ) -> None:
        self.provider = provider
        self.prompts = prompts
        self.config = config

    @property
    def prompt_text(self) -> str:
        return self.prompts.combined(self.role_name)

    def _parts(self, labeled_images: Sequence[tuple[str, str | Path]]) -> list[PromptPart]:
        parts: list[PromptPart] = [PromptPart("text", self.prompt_text, "instructions")]
        for label, path in labeled_images:
            parts.append(PromptPart("text", label, "image_label"))
            parts.append(PromptPart("image", Path(path), label))
        return parts

    def _call_record(
        self,
        response: ProviderResponse,
        parts: Sequence[PromptPart],
        *,
        failed_output_attempts: Sequence[dict[str, Any]] | None = None,
    ) -> AgentCallRecord:
        return AgentCallRecord(
            agent=self.role_name,
            prompt_text=self.prompt_text,
            request_parts=request_parts_manifest(parts),
            raw_response=response.text,
            provider_response=response,
            failed_output_attempts=list(failed_output_attempts or ()),
        )

    @staticmethod
    def ensure_renderable(route: tuple[int, ...], allowed_node_ids: set[int]) -> None:
        unknown = sorted({node for node in route if node not in allowed_node_ids})
        if unknown:
            raise ModelOutputError(f"Bilinmeyen node ID: {unknown}")
