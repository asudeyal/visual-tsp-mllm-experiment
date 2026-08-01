"""Sağlayıcıdan bağımsız görsel model arayüzü."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypeVar

from src.problem_instance import ProblemInstance
from src.request_control import (
    RequestController,
    extract_provider_timing,
)


T = TypeVar("T")


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

    _request_controller: RequestController | None = None

    @property
    def model_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "alias": self.model_alias,
            "requested_name": self.resolved_model,
            "capabilities": {
                "supports_vision": (
                    self.capabilities.supports_vision
                ),
                "supports_multiple_images": (
                    self.capabilities.supports_multiple_images
                ),
                "supports_native_multiple_choices": (
                    self.capabilities
                    .supports_native_multiple_choices
                ),
                "max_images_per_request": (
                    self.capabilities.max_images_per_request
                ),
                "max_native_choices": (
                    self.capabilities.max_native_choices
                ),
            },
        }

    def configure_request_controller(
        self,
        controller: RequestController | None,
    ) -> ProviderAdapter:
        """
        Provider'ın gerçek API çağrılarını ortak kontrol katmanına bağlar.

        None verilirse provider önceki doğrudan çalışma biçimini korur.
        """

        self._request_controller = controller
        return self

    @staticmethod
    def _enrich_api_call(
        call: dict[str, Any],
        *,
        request_control: dict[str, Any] | None,
    ) -> None:
        if request_control is not None:
            call["request_control"] = dict(request_control)

        call["provider_timing"] = extract_provider_timing(
            call.get("usage"),
            api_call_wall_seconds=call.get(
                "api_call_wall_seconds"
            ),
        )

    def _decorate_result(
        self,
        value: T,
        *,
        request_control: dict[str, Any] | None,
    ) -> T:
        seen: set[int] = set()
        calls: list[dict[str, Any]] = []

        primary_call = getattr(value, "api_call", None)

        if isinstance(primary_call, dict):
            calls.append(primary_call)

        individual_calls = getattr(value, "api_calls", None)

        if isinstance(individual_calls, list):
            calls.extend(
                call
                for call in individual_calls
                if isinstance(call, dict)
            )

        for call in calls:
            identity = id(call)

            if identity in seen:
                continue

            seen.add(identity)

            self._enrich_api_call(
                call,
                request_control=request_control,
            )

        return value

    def _decorate_error(
        self,
        error: Exception,
        *,
        request_control: dict[str, Any],
    ) -> None:
        seen: set[int] = set()

        for attribute_name in (
            "provider_call_record",
            "gemini_call_record",
            "groq_call_record",
            "openrouter_call_record",
        ):
            call = getattr(
                error,
                attribute_name,
                None,
            )

            if not isinstance(call, dict):
                continue

            identity = id(call)

            if identity in seen:
                continue

            seen.add(identity)

            self._enrich_api_call(
                call,
                request_control=request_control,
            )

        for attribute_name in (
            "provider_call_records",
            "openrouter_call_records",
        ):
            calls = getattr(
                error,
                attribute_name,
                None,
            )

            if not isinstance(calls, list):
                continue

            for call in calls:
                if not isinstance(call, dict):
                    continue

                identity = id(call)

                if identity in seen:
                    continue

                seen.add(identity)

                self._enrich_api_call(
                    call,
                    request_control=request_control,
                )

        try:
            setattr(
                error,
                "request_control_report",
                request_control,
            )
        except Exception:
            pass

    def _execute_request(
        self,
        operation: Callable[[], T],
        *,
        label: str,
    ) -> T:
        """
        Tek bir gerçek API isteğini çalıştırır.

        Retry uygulanıyorsa aynı operation yeniden çalıştırılır. Bu metot
        aday grubunun tamamını değil, tek HTTP isteğini sarmalamalıdır.
        """

        controller = self._request_controller

        if controller is None:
            value = operation()

            return self._decorate_result(
                value,
                request_control=None,
            )

        outcome = controller.execute(
            operation,
            label=label,
        )
        report = outcome.to_dict()

        if outcome.success:
            value = outcome.unwrap()

            return self._decorate_result(
                value,
                request_control=report,
            )

        if outcome.error is not None:
            self._decorate_error(
                outcome.error,
                request_control=report,
            )

        return outcome.unwrap()

    def validate_candidate_count(
        self,
        candidate_count: int,
    ) -> None:
        if candidate_count < 1:
            raise ValueError(
                "candidate-count en az 1 olmalıdır."
            )

        maximum = self.capabilities.max_images_per_request

        if (
            maximum is not None
            and candidate_count > maximum
        ):
            raise ValueError(
                f"{self.provider_id}/{self.model_alias} "
                "scorer çağrısında "
                f"en fazla {maximum} görsel destekliyor; "
                f"candidate-count={candidate_count} "
                "kullanılamaz."
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