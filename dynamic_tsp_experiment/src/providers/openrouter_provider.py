"""OpenRouter sağlayıcı adaptörü."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src import openrouter
from src.problem_instance import ProblemInstance
from src.providers.base import (
    ProviderAdapter,
    ProviderCandidatesResult,
    ProviderCapabilities,
    ProviderTextResult,
)


class OpenRouterProvider(ProviderAdapter):
    provider_id = "openrouter"
    default_candidate_strategy = "independent_calls"
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=True,
        max_images_per_request=7,
        max_native_choices=7,
    )

    def __init__(self, model: str) -> None:
        self.model_alias = model
        self.resolved_model = (
            openrouter.OPENROUTER_MODELS.get(model, model)
        )

    def request_route(
        self,
        image_path: Path,
        *,
        prompt: str,
        temperature: float,
        phase: str,
    ) -> ProviderTextResult:
        result = openrouter.request_route(
            image_path,
            prompt=prompt,
            model=self.resolved_model,
            temperature=temperature,
            max_tokens=8192,
            reasoning_effort="none",
            phase=phase,
        )
        return ProviderTextResult(result.text, result.api_call)

    def request_candidates(
        self,
        image_path: Path,
        *,
        problem: ProblemInstance,
        candidate_count: int,
        temperature: float,
        strategy: str,
    ) -> ProviderCandidatesResult:
        self.validate_candidate_count(candidate_count)
        if strategy == "auto":
            strategy = "independent_calls"
        result = openrouter.request_candidates(
            image_path,
            problem=problem,
            candidate_count=candidate_count,
            model=self.resolved_model,
            temperature=temperature,
            strategy=strategy,
        )
        return ProviderCandidatesResult(
            result.texts,
            result.api_call,
            result.api_calls,
        )

    def request_scorer(
        self,
        image_paths: Sequence[Path],
        *,
        problem: ProblemInstance,
        image_ids: Sequence[int],
    ) -> ProviderTextResult:
        result = openrouter.request_scorer(
            list(image_paths),
            problem=problem,
            image_ids=list(image_ids),
            model=self.resolved_model,
        )
        return ProviderTextResult(result.text, result.api_call)
