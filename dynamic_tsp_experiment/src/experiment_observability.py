from __future__ import annotations

import argparse
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from src.request_control import (
    RequestController,
    RetryPolicy,
    WaitTracker,
)
from src.runtime_metrics import ResourceSampler
from src.solution_tracking import EarlyStopPolicy


@dataclass(frozen=True)
class ObservabilitySettings:
    profile_resources: bool = True
    resource_sample_interval_seconds: float = 0.5
    minimum_request_interval_seconds: float = 0.0
    max_retries: int = 2
    retry_base_delay_seconds: float = 2.0
    retry_maximum_delay_seconds: float = 60.0
    early_stop_enabled: bool = True
    early_stop_gap_percent: float = 1.0

    def __post_init__(self) -> None:
        numeric_nonnegative = {
            "minimum_request_interval_seconds": (
                self.minimum_request_interval_seconds
            ),
            "retry_base_delay_seconds": (
                self.retry_base_delay_seconds
            ),
            "retry_maximum_delay_seconds": (
                self.retry_maximum_delay_seconds
            ),
            "early_stop_gap_percent": (
                self.early_stop_gap_percent
            ),
        }

        for name, value in numeric_nonnegative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} sonlu ve negatif olmayan "
                    "bir sayı olmalıdır."
                )

        if (
            not math.isfinite(
                self.resource_sample_interval_seconds
            )
            or self.resource_sample_interval_seconds <= 0
        ):
            raise ValueError(
                "resource_sample_interval_seconds "
                "sıfırdan büyük olmalıdır."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries negatif olamaz."
            )

        if (
            self.retry_maximum_delay_seconds
            < self.retry_base_delay_seconds
        ):
            raise ValueError(
                "retry_maximum_delay_seconds, "
                "retry_base_delay_seconds değerinden "
                "küçük olamaz."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_resources": self.profile_resources,
            "resource_sample_interval_seconds": (
                self.resource_sample_interval_seconds
            ),
            "minimum_request_interval_seconds": (
                self.minimum_request_interval_seconds
            ),
            "max_retries": self.max_retries,
            "retry_base_delay_seconds": (
                self.retry_base_delay_seconds
            ),
            "retry_maximum_delay_seconds": (
                self.retry_maximum_delay_seconds
            ),
            "early_stop_enabled": self.early_stop_enabled,
            "early_stop_gap_percent": (
                self.early_stop_gap_percent
            ),
        }


def add_observability_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_early_stop: bool,
) -> None:
    parser.add_argument(
        "--profile-resources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "CPU, RAM ve varsa NVIDIA GPU kullanımını ölçer. "
            "--no-profile-resources ile kapatılabilir."
        ),
    )
    parser.add_argument(
        "--resource-sample-interval-seconds",
        type=float,
        default=0.5,
        help="Kaynak kullanımının örnekleme aralığı.",
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=0.0,
        help=(
            "Ardışık API istek başlangıçları arasındaki "
            "minimum kontrollü süre."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help=(
            "429/503/504 sonrasında yapılabilecek "
            "azami ek deneme sayısı."
        ),
    )
    parser.add_argument(
        "--retry-base-delay-seconds",
        type=float,
        default=2.0,
        help=(
            "Sağlayıcı retry süresi vermediğinde "
            "kullanılan ilk backoff süresi."
        ),
    )
    parser.add_argument(
        "--retry-maximum-delay-seconds",
        type=float,
        default=60.0,
        help="Tek retry için azami backoff süresi.",
    )

    if include_early_stop:
        parser.add_argument(
            "--early-stop-gap-percent",
            type=float,
            default=1.0,
            help=(
                "Kanıtlanmış optimum referansında sistem GBest "
                "bu gap değerine ulaştığında durur."
            ),
        )
        parser.add_argument(
            "--disable-early-stop",
            action="store_true",
            help="Gap tabanlı erken durdurmayı kapatır.",
        )


def settings_from_args(
    args: argparse.Namespace,
    *,
    include_early_stop: bool,
) -> ObservabilitySettings:
    return ObservabilitySettings(
        profile_resources=bool(args.profile_resources),
        resource_sample_interval_seconds=float(
            args.resource_sample_interval_seconds
        ),
        minimum_request_interval_seconds=float(
            args.request_interval_seconds
        ),
        max_retries=int(args.max_retries),
        retry_base_delay_seconds=float(
            args.retry_base_delay_seconds
        ),
        retry_maximum_delay_seconds=float(
            args.retry_maximum_delay_seconds
        ),
        early_stop_enabled=(
            not bool(
                getattr(
                    args,
                    "disable_early_stop",
                    False,
                )
            )
            if include_early_stop
            else False
        ),
        early_stop_gap_percent=(
            float(
                getattr(
                    args,
                    "early_stop_gap_percent",
                    1.0,
                )
            )
            if include_early_stop
            else 1.0
        ),
    )


class ExperimentObservability:
    def __init__(
        self,
        settings: ObservabilitySettings,
        *,
        wait_tracker: WaitTracker | None = None,
        resource_sampler: ResourceSampler | None = None,
    ) -> None:
        self.settings = settings

        self.wait_tracker = wait_tracker or WaitTracker()

        self.request_controller = RequestController(
            retry_policy=RetryPolicy(
                max_retries=settings.max_retries,
                base_delay_seconds=(
                    settings.retry_base_delay_seconds
                ),
                maximum_delay_seconds=(
                    settings.retry_maximum_delay_seconds
                ),
            ),
            minimum_request_interval_seconds=(
                settings.minimum_request_interval_seconds
            ),
            wait_tracker=self.wait_tracker,
        )

        if resource_sampler is not None:
            self.resource_sampler = resource_sampler
        elif settings.profile_resources:
            self.resource_sampler = ResourceSampler(
                interval_seconds=(
                    settings.resource_sample_interval_seconds
                )
            )
        else:
            self.resource_sampler = None

        self._started = False
        self._stopped = False
        self._resource_summary: dict[str, Any] | None = None

    def start(self) -> ExperimentObservability:
        if self._started:
            return self

        if self._stopped:
            raise RuntimeError(
                "Durdurulmuş gözlem oturumu yeniden başlatılamaz."
            )

        if self.resource_sampler is not None:
            self.resource_sampler.start()

        self._started = True
        return self

    @contextmanager
    def phase(self, phase_name: str) -> Iterator[None]:
        if self.resource_sampler is None or not self._started:
            yield
            return

        with self.resource_sampler.phase(phase_name):
            yield

    def early_stop_policy(self) -> EarlyStopPolicy:
        return EarlyStopPolicy(
            enabled=self.settings.early_stop_enabled,
            threshold_percent=(
                self.settings.early_stop_gap_percent
            ),
        )

    def stop(self) -> dict[str, Any]:
        if self._stopped:
            return self.summary()

        if self.resource_sampler is not None:
            self._resource_summary = (
                self.resource_sampler.stop()
            )
        else:
            self._resource_summary = {
                "enabled": False,
                "reason": "disabled_by_configuration",
                "sample_count": 0,
            }

        self._started = False
        self._stopped = True
        return self.summary()

    def summary(self) -> dict[str, Any]:
        if self._resource_summary is not None:
            resources = self._resource_summary
        elif self.resource_sampler is not None:
            resources = self.resource_sampler.summary()
        else:
            resources = {
                "enabled": False,
                "reason": "disabled_by_configuration",
                "sample_count": 0,
            }

        request_control = self.request_controller.summary()

        return {
            "settings": self.settings.to_dict(),
            "controlled_waits": request_control["waits"],
            "request_control": request_control,
            "resources": resources,
        }

    def __enter__(self) -> ExperimentObservability:
        return self.start()

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.stop()