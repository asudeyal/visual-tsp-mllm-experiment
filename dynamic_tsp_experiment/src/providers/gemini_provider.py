"""Google Gemini sağlayıcı adaptörü."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src import gemini
from src.metrics import summarize_api_calls
from src.problem_instance import ProblemInstance
from src.providers.base import (
    ProviderAdapter,
    ProviderCandidatesResult,
    ProviderCapabilities,
    ProviderTextResult,
)


class GeminiProvider(ProviderAdapter):
    provider_id = "gemini"
    default_candidate_strategy = (
        "native_multiple_choices"
    )
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=True,
        max_images_per_request=7,
        max_native_choices=7,
    )

    def __init__(self, model: str) -> None:
        self.model_alias = model
        self.resolved_model = model

    def request_route(
        self,
        image_path: Path,
        *,
        prompt: str,
        temperature: float,
        phase: str,
    ) -> ProviderTextResult:
        result = self._execute_request(
            lambda: gemini.request_route(
                image_path,
                prompt=prompt,
                model=self.resolved_model,
                temperature=temperature,
                phase=phase,
            ),
            label=f"gemini:{phase}",
        )

        return ProviderTextResult(
            result.text,
            result.api_call,
        )

    def _independent_candidates(
        self,
        image_path: Path,
        *,
        problem: ProblemInstance,
        candidate_count: int,
        temperature: float,
    ) -> ProviderCandidatesResult:
        texts: list[str] = []
        calls: list[dict[str, Any]] = []

        for candidate_id in range(
            1,
            candidate_count + 1,
        ):
            result = self._execute_request(
                lambda: gemini.request_candidates(
                    image_path,
                    problem=problem,
                    candidate_count=1,
                    model=self.resolved_model,
                    temperature=temperature,
                ),
                label=(
                    "gemini:"
                    "critic_candidate_generation_"
                    f"{candidate_id:02d}"
                ),
            )

            texts.extend(result.texts)
            calls.append(result.api_call)

        summary = summarize_api_calls(calls)

        aggregate: dict[str, Any] = {
            "phase": "critic_candidate_generation",
            "provider": self.provider_id,
            "model": self.resolved_model,
            "temperature": temperature,
            "success": True,
            "strategy": "independent_calls",
            "http_request_count": len(calls),
            "api_call_wall_seconds": summary[
                "total_api_call_wall_seconds"
            ],
            "requested_candidate_count": (
                candidate_count
            ),
            "returned_candidate_count": len(texts),
            "usage": {
                "total_token_count": summary[
                    "total_token_count"
                ],
            },
        }

        return ProviderCandidatesResult(
            texts,
            aggregate,
            calls,
        )

    def request_candidates(
        self,
        image_path: Path,
        *,
        problem: ProblemInstance,
        candidate_count: int,
        temperature: float,
        strategy: str,
    ) -> ProviderCandidatesResult:
        self.validate_candidate_count(
            candidate_count
        )

        if strategy == "auto":
            strategy = "native_multiple_choices"

        if strategy == "independent_calls":
            return self._independent_candidates(
                image_path,
                problem=problem,
                candidate_count=candidate_count,
                temperature=temperature,
            )

        if strategy != "native_multiple_choices":
            raise ValueError(
                f"Bilinmeyen aday stratejisi: {strategy}"
            )

        result = self._execute_request(
            lambda: gemini.request_candidates(
                image_path,
                problem=problem,
                candidate_count=candidate_count,
                model=self.resolved_model,
                temperature=temperature,
            ),
            label=(
                "gemini:"
                "critic_candidate_generation_native"
            ),
        )

        return ProviderCandidatesResult(
            result.texts,
            result.api_call,
            [result.api_call],
        )

    def request_scorer(
        self,
        image_paths: Sequence[Path],
        *,
        problem: ProblemInstance,
        image_ids: Sequence[int],
    ) -> ProviderTextResult:
        result = self._execute_request(
            lambda: gemini.request_scorer(
                image_paths,
                problem=problem,
                image_ids=image_ids,
                model=self.resolved_model,
            ),
            label="gemini:visual_scorer",
        )

        return ProviderTextResult(
            result.text,
            result.api_call,
        )