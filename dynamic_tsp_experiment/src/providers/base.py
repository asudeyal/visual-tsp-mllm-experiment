"""Sağlayıcıdan bağımsız görsel model arayüzü."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.problem_instance import ProblemInstance


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_vision: bool
    supports_multiple_images: bool
    supports_native_multiple_choices: bool
    max_images_per_request: int | None
    max_native_choices: int


@dataclass(frozen=True)
class ProviderTextResult:
    text: str
    api_call: dict[str, Any]


@dataclass(frozen=True)
class ProviderCandidatesResult:
    texts: list[str]
    api_call: dict[str, Any]
    api_calls: list[dict[str, Any]]


class ProviderAdapter(ABC):
    """Zero-shot, critic ve scorer için ortak sağlayıcı sözleşmesi."""

    provider_id: str
    model_alias: str
    resolved_model: str
    capabilities: ProviderCapabilities
    default_candidate_strategy: str

    @property
    def model_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "alias": self.model_alias,
            "requested_name": self.resolved_model,
            "capabilities": {
                "supports_vision": self.capabilities.supports_vision,
                "supports_multiple_images": (
                    self.capabilities.supports_multiple_images
                ),
                "supports_native_multiple_choices": (
                    self.capabilities.supports_native_multiple_choices
                ),
                "max_images_per_request": (
                    self.capabilities.max_images_per_request
                ),
                "max_native_choices": (
                    self.capabilities.max_native_choices
                ),
            },
        }

    def validate_candidate_count(self, candidate_count: int) -> None:
        if candidate_count < 1:
            raise ValueError("candidate-count en az 1 olmalıdır.")
        maximum = self.capabilities.max_images_per_request
        if maximum is not None and candidate_count > maximum:
            raise ValueError(
                f"{self.provider_id}/{self.model_alias} scorer çağrısında "
                f"en fazla {maximum} görsel destekliyor; "
                f"candidate-count={candidate_count} kullanılamaz."
            )

    @abstractmethod
    def request_route(
        self,
        image_path: Path,
        *,
        prompt: str,
        temperature: float,
        phase: str,
    ) -> ProviderTextResult:
        raise NotImplementedError

    @abstractmethod
    def request_candidates(
        self,
        image_path: Path,
        *,
        problem: ProblemInstance,
        candidate_count: int,
        temperature: float,
        strategy: str,
    ) -> ProviderCandidatesResult:
        raise NotImplementedError

    @abstractmethod
    def request_scorer(
        self,
        image_paths: Sequence[Path],
        *,
        problem: ProblemInstance,
        image_ids: Sequence[int],
    ) -> ProviderTextResult:
        raise NotImplementedError
