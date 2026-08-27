"""Provider-independent multimodal interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Sequence

from ..schemas import PromptPart, ProviderCapabilities, ProviderManyResponse, ProviderResponse


class ProviderAdapter(ABC):
    provider_id: str
    capabilities: ProviderCapabilities
    # Some SDKs (currently Gemini's google-genai client) already retry transient
    # transport/server errors internally. In that case the shared adapter must
    # not wrap the same request in a second retry loop.
    sdk_managed_retries: bool = False

    def __init__(self, model: str, *, timeout_seconds: int = 120, request_retries: int = 3) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.request_retries = request_retries

    def _effective_temperature(self, temperature: float | None) -> float | None:
        return temperature if self.capabilities.supports_temperature else None

    def _effective_thinking_level(self, thinking_level: str | None) -> str | None:
        return thinking_level if self.capabilities.supports_thinking_level else None

    def generate(
        self,
        parts: Sequence[PromptPart],
        *,
        phase: str,
        temperature: float | None = None,
        thinking_level: str | None = None,
    ) -> ProviderResponse:
        last_error: Exception | None = None
        attempts = 1 if self.sdk_managed_retries else self.request_retries
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                response = self._generate_once(
                    parts,
                    phase=phase,
                    temperature=self._effective_temperature(temperature),
                    thinking_level=self._effective_thinking_level(thinking_level),
                )
                if response.latency_seconds is None:
                    response = replace(response, latency_seconds=time.perf_counter() - started)
                response.raw_metadata.setdefault("request_attempt", attempt)
                response.raw_metadata.setdefault(
                    "retry_owner",
                    "provider_sdk" if self.sdk_managed_retries else "avma_provider_adapter",
                )
                return response
            except Exception as exc:  # provider-specific HTTP/SDK errors
                last_error = exc
                if attempt >= attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error

    def generate_many(
        self,
        parts: Sequence[PromptPart],
        *,
        count: int,
        phase: str,
        temperature: float | None = None,
        thinking_level: str | None = None,
        strategy: str = "auto",
    ) -> ProviderManyResponse:
        if count < 1:
            raise ValueError("count en az 1 olmalıdır")
        if strategy not in {"auto", "independent_calls", "native_multiple_choices"}:
            raise ValueError(f"Bilinmeyen candidate strategy: {strategy}")

        use_native = strategy == "native_multiple_choices" or (
            strategy == "auto" and self.capabilities.supports_native_multiple_choices
        )
        if use_native:
            if not self.capabilities.supports_native_multiple_choices:
                raise ValueError(f"{self.provider_id} native multiple choices desteklemiyor")
            if count > self.capabilities.max_native_choices:
                raise ValueError(
                    f"{self.provider_id} en fazla {self.capabilities.max_native_choices} native choice destekliyor"
                )
            started = time.perf_counter()
            result = self._generate_many_native(
                parts,
                count=count,
                phase=phase,
                temperature=self._effective_temperature(temperature),
                thinking_level=self._effective_thinking_level(thinking_level),
            )
            if result.responses:
                first = result.responses[0]
                if first.latency_seconds is None:
                    result.responses[0] = replace(
                        first,
                        latency_seconds=time.perf_counter() - started,
                    )
            return result

        responses = [
            self.generate(
                parts,
                phase=f"{phase}_{index:02d}",
                temperature=temperature,
                thinking_level=thinking_level,
            )
            for index in range(1, count + 1)
        ]
        return ProviderManyResponse(responses=responses, strategy="independent_calls")

    def _generate_many_native(
        self,
        parts: Sequence[PromptPart],
        *,
        count: int,
        phase: str,
        temperature: float | None,
        thinking_level: str | None,
    ) -> ProviderManyResponse:
        raise NotImplementedError

    @abstractmethod
    def _generate_once(
        self,
        parts: Sequence[PromptPart],
        *,
        phase: str,
        temperature: float | None,
        thinking_level: str | None,
    ) -> ProviderResponse:
        raise NotImplementedError
