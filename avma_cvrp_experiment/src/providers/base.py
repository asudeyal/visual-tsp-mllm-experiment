"""Provider-independent multimodal interface."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
        self.request_delay_seconds = 0.0
        self._last_request_started_at: float | None = None

    def configure_request_delay(self, seconds: float) -> None:
        self.request_delay_seconds = max(0.0, float(seconds))

    def _effective_temperature(self, temperature: float | None) -> float | None:
        return temperature if self.capabilities.supports_temperature else None

    def _effective_thinking_level(self, thinking_level: str | None) -> str | None:
        return thinking_level if self.capabilities.supports_thinking_level else None

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        for attribute in ("status_code", "code"):
            value = getattr(error, attribute, None)
            if value is not None and not callable(value):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        value = headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _is_retryable_error(cls, error: Exception) -> bool:
        status = cls._status_code(error)
        if status is not None:
            return status in {408, 429} or 500 <= status <= 599
        # Transport-level timeouts/connection errors are surfaced immediately.
        # Only explicit transient HTTP statuses use automatic retry/backoff.
        return False

    @classmethod
    def _retry_wait_seconds(cls, error: Exception, failed_attempt: int) -> float:
        retry_after = cls._retry_after_seconds(error)
        if retry_after is not None:
            return retry_after
        return float(min(30 * (2 ** max(0, failed_attempt - 1)), 120))

    def _wait_for_request_slot(self) -> float:
        waited = 0.0
        now = time.monotonic()
        if self.request_delay_seconds > 0 and self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            waited = max(0.0, self.request_delay_seconds - elapsed)
            if waited > 0:
                time.sleep(waited)
                now = time.monotonic()
        self._last_request_started_at = now
        return waited

    @staticmethod
    def _attach_wait_metadata(
        error: Exception,
        *,
        request_delay_wait_seconds: float,
        retry_backoff_wait_seconds: float,
    ) -> None:
        metadata = {
            "request_delay_wait_seconds": request_delay_wait_seconds,
            "retry_backoff_wait_seconds": retry_backoff_wait_seconds,
            "provider_wait_seconds": request_delay_wait_seconds + retry_backoff_wait_seconds,
        }
        try:
            setattr(error, "_avma_wait_metadata", metadata)
        except Exception:
            pass

    def generate(
        self,
        parts: Sequence[PromptPart],
        *,
        phase: str,
        temperature: float | None = None,
        thinking_level: str | None = None,
    ) -> ProviderResponse:
        last_error: Exception | None = None
        request_delay_wait_seconds = 0.0
        retry_backoff_wait_seconds = 0.0
        attempts = 1 if self.sdk_managed_retries else self.request_retries
        for attempt in range(1, attempts + 1):
            request_delay_wait_seconds += self._wait_for_request_slot()
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
                response.raw_metadata.setdefault(
                    "request_delay_wait_seconds", request_delay_wait_seconds
                )
                response.raw_metadata.setdefault(
                    "retry_backoff_wait_seconds", retry_backoff_wait_seconds
                )
                response.raw_metadata.setdefault(
                    "provider_wait_seconds",
                    request_delay_wait_seconds + retry_backoff_wait_seconds,
                )
                return response
            except Exception as exc:  # provider-specific HTTP/SDK errors
                last_error = exc
                if attempt >= attempts or not self._is_retryable_error(exc):
                    self._attach_wait_metadata(
                        exc,
                        request_delay_wait_seconds=request_delay_wait_seconds,
                        retry_backoff_wait_seconds=retry_backoff_wait_seconds,
                    )
                    raise
                wait_seconds = self._retry_wait_seconds(exc, attempt)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                    retry_backoff_wait_seconds += wait_seconds
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
