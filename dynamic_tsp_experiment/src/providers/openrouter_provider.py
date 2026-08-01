"""OpenRouter sağlayıcı adaptörü."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src import openrouter
from src.problem_instance import ProblemInstance
from src.providers.base import (
    ProviderAdapter,
    ProviderCandidatesResult,
    ProviderCapabilities,
    ProviderTextResult,
)


def _summed_usage(
    calls: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fields = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "total_token_count",
        "cost",
    )

    result: dict[str, Any] = {}

    for field in fields:
        known_values: list[float] = []

        for call in calls:
            value = (
                call.get("usage") or {}
            ).get(field)

            if isinstance(value, bool):
                continue

            if isinstance(value, (int, float)):
                known_values.append(float(value))

        result[field] = (
            sum(known_values)
            if known_values
            else None
        )

    return result


def _aggregate_independent_calls(
    calls: Sequence[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    candidate_count: int,
    returned_candidate_count: int,
) -> dict[str, Any]:
    return {
        "phase": "critic_candidate_generation",
        "provider": "openrouter",
        "model": model,
        "temperature": temperature,
        "success": True,
        "strategy": "independent_calls",
        "http_request_count": len(calls),
        "started_at_utc": (
            calls[0].get("started_at_utc")
            if calls
            else None
        ),
        "finished_at_utc": (
            calls[-1].get("finished_at_utc")
            if calls
            else None
        ),
        "api_call_wall_seconds": sum(
            float(
                call.get(
                    "api_call_wall_seconds",
                    0.0,
                )
                or 0.0
            )
            for call in calls
        ),
        "input_image_count": sum(
            int(
                call.get(
                    "input_image_count",
                    0,
                )
                or 0
            )
            for call in calls
        ),
        "input_image_bytes": sum(
            int(
                call.get(
                    "input_image_bytes",
                    0,
                )
                or 0
            )
            for call in calls
        ),
        "requested_candidate_count": (
            candidate_count
        ),
        "returned_candidate_count": (
            returned_candidate_count
        ),
        "response_models": sorted(
            {
                str(call["response_model"])
                for call in calls
                if call.get("response_model")
            }
        ),
        "routed_providers": sorted(
            {
                str(call["routed_provider"])
                for call in calls
                if call.get("routed_provider")
            }
        ),
        "finish_reasons": [
            call.get("finish_reason")
            for call in calls
        ],
        "usage": _summed_usage(calls),
    }


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
            openrouter.OPENROUTER_MODELS.get(
                model,
                model,
            )
        )

    def request_route(
        self,
        image_path: Path,
        *,
        prompt: str,
        temperature: float,
        phase: str,
    ) -> ProviderTextResult:
        result = self._execute_request(
            lambda: openrouter.request_route(
                image_path,
                prompt=prompt,
                model=self.resolved_model,
                temperature=temperature,
                max_tokens=8192,
                reasoning_effort="none",
                phase=phase,
            ),
            label=f"openrouter:{phase}",
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
            try:
                result = self._execute_request(
                    lambda: (
                        openrouter.request_candidates(
                            image_path,
                            problem=problem,
                            candidate_count=1,
                            model=self.resolved_model,
                            temperature=temperature,
                            strategy=(
                                "native_multiple_choices"
                            ),
                        )
                    ),
                    label=(
                        "openrouter:"
                        "critic_candidate_generation_"
                        f"{candidate_id:02d}"
                    ),
                )
            except Exception as exc:
                failed_records = getattr(
                    exc,
                    "openrouter_call_records",
                    None,
                )

                if not isinstance(
                    failed_records,
                    list,
                ):
                    single_record = getattr(
                        exc,
                        "openrouter_call_record",
                        None,
                    )
                    failed_records = (
                        [single_record]
                        if isinstance(
                            single_record,
                            dict,
                        )
                        else []
                    )

                try:
                    setattr(
                        exc,
                        "provider_call_records",
                        [
                            *calls,
                            *failed_records,
                        ],
                    )
                except Exception:
                    pass

                raise

            texts.extend(result.texts)
            calls.extend(result.api_calls)

        aggregate = _aggregate_independent_calls(
            calls,
            model=self.resolved_model,
            temperature=temperature,
            candidate_count=candidate_count,
            returned_candidate_count=len(texts),
        )

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
            strategy = "independent_calls"

        if strategy == "independent_calls":
            return self._independent_candidates(
                image_path,
                problem=problem,
                candidate_count=candidate_count,
                temperature=temperature,
            )

        if strategy != "native_multiple_choices":
            raise ValueError(
                "Bilinmeyen critic aday stratejisi: "
                f"{strategy}"
            )

        result = self._execute_request(
            lambda: openrouter.request_candidates(
                image_path,
                problem=problem,
                candidate_count=candidate_count,
                model=self.resolved_model,
                temperature=temperature,
                strategy=(
                    "native_multiple_choices"
                ),
            ),
            label=(
                "openrouter:"
                "critic_candidate_generation_native"
            ),
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
        result = self._execute_request(
            lambda: openrouter.request_scorer(
                list(image_paths),
                problem=problem,
                image_ids=list(image_ids),
                model=self.resolved_model,
            ),
            label="openrouter:visual_scorer",
        )

        return ProviderTextResult(
            result.text,
            result.api_call,
        )