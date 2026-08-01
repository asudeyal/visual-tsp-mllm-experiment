from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


T = TypeVar("T")

DELIBERATE_DELAY = "deliberate_delay"
RATE_LIMIT_BACKOFF = "rate_limit_backoff"

CONTROLLED_WAIT_TYPES = (
    DELIBERATE_DELAY,
    RATE_LIMIT_BACKOFF,
)


def _finite_nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
        return None

    converted = float(value)

    if not math.isfinite(converted) or converted < 0:
        return None

    return converted


def _safe_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    model_dump = getattr(value, "model_dump", None)

    if callable(model_dump):
        try:
            result = model_dump()

            if isinstance(result, Mapping):
                return result
        except Exception:
            pass

    dictionary = getattr(value, "__dict__", None)

    if isinstance(dictionary, Mapping):
        return dictionary

    return None


def _case_insensitive_get(
    mapping: Mapping[str, Any],
    key: str,
) -> Any:
    expected = key.casefold()

    for current_key, current_value in mapping.items():
        if str(current_key).casefold() == expected:
            return current_value

    return None


def _exception_text(error: BaseException) -> str:
    message = str(error).strip()

    if message:
        return message

    return type(error).__name__


def extract_status_code(error: BaseException) -> int | None:
    """
    Farklı API istemcilerindeki HTTP durum kodunu ortak biçimde bulur.
    """

    candidates: list[Any] = []

    for attribute_name in (
        "status_code",
        "http_status",
        "status",
        "code",
    ):
        candidates.append(getattr(error, attribute_name, None))

    response = getattr(error, "response", None)

    if response is not None:
        for attribute_name in (
            "status_code",
            "status",
            "code",
        ):
            candidates.append(
                getattr(response, attribute_name, None)
            )

        response_mapping = _safe_mapping(response)

        if response_mapping is not None:
            for key in (
                "status_code",
                "status",
                "code",
            ):
                candidates.append(
                    _case_insensitive_get(response_mapping, key)
                )

    error_mapping = _safe_mapping(error)

    if error_mapping is not None:
        for key in (
            "status_code",
            "status",
            "code",
        ):
            candidates.append(
                _case_insensitive_get(error_mapping, key)
            )

    for candidate in candidates:
        if isinstance(candidate, bool):
            continue

        if isinstance(candidate, int) and 100 <= candidate <= 599:
            return candidate

        if isinstance(candidate, str) and candidate.isdigit():
            converted = int(candidate)

            if 100 <= converted <= 599:
                return converted

    message = _exception_text(error)

    patterns = (
        r"\bHTTP\s+([1-5]\d{2})\b",
        r"\b([1-5]\d{2})\s+RESOURCE_EXHAUSTED\b",
        r"""["']?code["']?\s*[:=]\s*([1-5]\d{2})\b""",
        r"""["']?status_code["']?\s*[:=]\s*([1-5]\d{2})\b""",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            message,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


def _headers_from_error(
    error: BaseException,
) -> Mapping[str, Any] | None:
    headers = getattr(error, "headers", None)
    headers_mapping = _safe_mapping(headers)

    if headers_mapping is not None:
        return headers_mapping

    response = getattr(error, "response", None)

    if response is not None:
        response_headers = getattr(response, "headers", None)
        headers_mapping = _safe_mapping(response_headers)

        if headers_mapping is not None:
            return headers_mapping

        response_mapping = _safe_mapping(response)

        if response_mapping is not None:
            nested_headers = _case_insensitive_get(
                response_mapping,
                "headers",
            )

            headers_mapping = _safe_mapping(nested_headers)

            if headers_mapping is not None:
                return headers_mapping

    return None


def _duration_to_seconds(
    number_text: str,
    unit_text: str | None,
) -> float | None:
    try:
        value = float(number_text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value) or value < 0:
        return None

    normalized_unit = (unit_text or "s").strip().casefold()

    if normalized_unit in (
        "ms",
        "millisecond",
        "milliseconds",
    ):
        return value / 1000.0

    return value


def extract_retry_after_seconds(
    error: BaseException,
) -> float | None:
    """
    Retry-After başlığı veya hata mesajındaki bekleme bilgisini bulur.
    """

    for attribute_name in (
        "retry_after_seconds",
        "retry_after",
        "retry_delay_seconds",
    ):
        value = getattr(error, attribute_name, None)

        converted = _finite_nonnegative_float(value)

        if converted is not None:
            return converted

    headers = _headers_from_error(error)

    if headers is not None:
        retry_after = _case_insensitive_get(
            headers,
            "retry-after",
        )

        if retry_after is not None:
            try:
                converted = float(str(retry_after).strip())
            except ValueError:
                converted = None

            if (
                converted is not None
                and math.isfinite(converted)
                and converted >= 0
            ):
                return converted

    message = _exception_text(error)

    patterns = (
        (
            r"please\s+retry\s+in\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(ms|milliseconds?|seconds?|s)?"
        ),
        (
            r"try\s+again\s+in\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(ms|milliseconds?|seconds?|s)?"
        ),
        (
            r"retryDelay"
            r"""["']?\s*[:=]\s*["']?"""
            r"(\d+(?:\.\d+)?)\s*"
            r"(ms|milliseconds?|seconds?|s)?"
        ),
        (
            r"retry[_\s-]*after"
            r"""["']?\s*[:=]\s*["']?"""
            r"(\d+(?:\.\d+)?)\s*"
            r"(ms|milliseconds?|seconds?|s)?"
        ),
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            message,
            flags=re.IGNORECASE,
        )

        if match:
            return _duration_to_seconds(
                match.group(1),
                match.group(2),
            )

    return None


def is_daily_quota_error(error: BaseException) -> bool:
    """
    Günlük kotanın birkaç saniye bekleyerek düzelmeyeceğini belirler.
    """

    message = _exception_text(error).casefold()

    markers = (
        "generaterequestsperday",
        "requestsperday",
        "requests per day",
        "requests-per-day",
        "per day per project",
        "daily quota",
        "daily request limit",
    )

    return any(marker in message for marker in markers)


def extract_provider_timing(
    usage: Any,
    *,
    api_call_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Groq gibi sağlayıcıların usage içinde verdiği sunucu sürelerini ayıklar.

    Sağlayıcı süre vermiyorsa available=False döndürür.
    """

    usage_mapping = _safe_mapping(usage)

    if usage_mapping is None:
        usage_mapping = {}

    raw_usage = _case_insensitive_get(
        usage_mapping,
        "raw",
    )
    raw_mapping = _safe_mapping(raw_usage)

    sources = [
        mapping
        for mapping in (
            raw_mapping,
            usage_mapping,
        )
        if mapping is not None
    ]

    def find_number(*keys: str) -> float | None:
        for source in sources:
            for key in keys:
                value = _case_insensitive_get(source, key)
                converted = _finite_nonnegative_float(value)

                if converted is not None:
                    return converted

        return None

    queue_seconds = find_number(
        "queue_time",
        "queue_seconds",
    )
    prompt_seconds = find_number(
        "prompt_time",
        "prompt_seconds",
    )
    completion_seconds = find_number(
        "completion_time",
        "completion_seconds",
    )
    provider_total_seconds = find_number(
        "total_time",
        "provider_total_seconds",
    )

    available = any(
        value is not None
        for value in (
            queue_seconds,
            prompt_seconds,
            completion_seconds,
            provider_total_seconds,
        )
    )

    normalized_api_wall = _finite_nonnegative_float(
        api_call_wall_seconds
    )

    estimated_overhead_seconds: float | None = None

    if (
        normalized_api_wall is not None
        and provider_total_seconds is not None
    ):
        estimated_overhead_seconds = max(
            0.0,
            normalized_api_wall - provider_total_seconds,
        )

    return {
        "available": available,
        "source": "provider_usage" if available else None,
        "provider_queue_seconds": queue_seconds,
        "provider_prompt_seconds": prompt_seconds,
        "provider_completion_seconds": completion_seconds,
        "provider_total_seconds": provider_total_seconds,
        "estimated_network_or_client_overhead_seconds": (
            estimated_overhead_seconds
        ),
    }


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay_seconds: float = 2.0
    maximum_delay_seconds: float = 60.0
    retryable_status_codes: tuple[int, ...] = (
        429,
        503,
        504,
    )

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(
                "max_retries negatif olamaz."
            )

        if self.base_delay_seconds < 0:
            raise ValueError(
                "base_delay_seconds negatif olamaz."
            )

        if self.maximum_delay_seconds < 0:
            raise ValueError(
                "maximum_delay_seconds negatif olamaz."
            )

        if (
            self.maximum_delay_seconds
            < self.base_delay_seconds
        ):
            raise ValueError(
                "maximum_delay_seconds, base_delay_seconds "
                "değerinden küçük olamaz."
            )


class WaitTracker:
    """
    Programın bilerek yaptığı beklemeleri türlerine göre kaydeder.
    """

    def __init__(
        self,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._sleeper = sleeper
        self._clock = clock
        self._started_at = clock()
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def wait(
        self,
        wait_type: str,
        seconds: float,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if wait_type not in CONTROLLED_WAIT_TYPES:
            raise ValueError(
                f"Desteklenmeyen bekleme türü: {wait_type}"
            )

        requested_seconds = _finite_nonnegative_float(seconds)

        if requested_seconds is None:
            raise ValueError(
                "Bekleme süresi sonlu ve negatif olmayan "
                "bir sayı olmalıdır."
            )

        if requested_seconds == 0:
            return None

        event_started_at = self._clock()
        relative_started_at = max(
            0.0,
            event_started_at - self._started_at,
        )

        self._sleeper(requested_seconds)

        actual_seconds = max(
            0.0,
            self._clock() - event_started_at,
        )

        event = {
            "type": wait_type,
            "reason": reason,
            "requested_seconds": requested_seconds,
            "actual_seconds": actual_seconds,
            "started_at_elapsed_seconds": relative_started_at,
            "metadata": dict(metadata or {}),
        }

        with self._lock:
            self._events.append(event)

        return dict(event)

    def snapshot(
        self,
        *,
        since_event_index: int = 0,
        include_events: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            events = [
                dict(event)
                for event in self._events[since_event_index:]
            ]

        deliberate_seconds = sum(
            float(event["actual_seconds"])
            for event in events
            if event["type"] == DELIBERATE_DELAY
        )
        backoff_seconds = sum(
            float(event["actual_seconds"])
            for event in events
            if event["type"] == RATE_LIMIT_BACKOFF
        )

        result = {
            "event_count": len(events),
            "deliberate_delay_seconds": deliberate_seconds,
            "rate_limit_backoff_seconds": backoff_seconds,
            "controlled_wait_seconds": (
                deliberate_seconds + backoff_seconds
            ),
        }

        if include_events:
            result["events"] = events

        return result


@dataclass
class RequestOutcome(Generic[T]):
    success: bool
    value: T | None
    error: Exception | None
    attempts: list[dict[str, Any]]
    waits: dict[str, Any]
    active_wall_seconds: float
    total_wall_seconds: float

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def retry_count(self) -> int:
        return max(0, self.attempt_count - 1)

    def unwrap(self) -> T:
        if self.success:
            return self.value  # type: ignore[return-value]

        if self.error is not None:
            raise self.error

        raise RuntimeError(
            "İstek başarısız oldu fakat hata bilgisi bulunamadı."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "attempt_count": self.attempt_count,
            "retry_count": self.retry_count,
            "attempts": [
                dict(attempt)
                for attempt in self.attempts
            ],
            "waits": dict(self.waits),
            "active_wall_seconds": self.active_wall_seconds,
            "total_wall_seconds": self.total_wall_seconds,
            "final_error": (
                {
                    "type": type(self.error).__name__,
                    "message": str(self.error),
                    "status_code": extract_status_code(self.error),
                }
                if self.error is not None
                else None
            ),
        }


class RequestController:
    """
    API çağrılarını minimum istek aralığı ve sınırlı retry ile çalıştırır.
    """

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        minimum_request_interval_seconds: float = 0.0,
        wait_tracker: WaitTracker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        minimum_interval = _finite_nonnegative_float(
            minimum_request_interval_seconds
        )

        if minimum_interval is None:
            raise ValueError(
                "minimum_request_interval_seconds geçersiz."
            )

        self.retry_policy = retry_policy or RetryPolicy()
        self.minimum_request_interval_seconds = minimum_interval
        self.wait_tracker = wait_tracker or WaitTracker(clock=clock)
        self._clock = clock

        self._last_request_started_at: float | None = None
        self._execution_reports: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _apply_request_spacing(
        self,
        *,
        label: str,
        attempt_number: int,
    ) -> None:
        with self._lock:
            last_started_at = self._last_request_started_at

        if last_started_at is None:
            return

        elapsed_since_previous_start = max(
            0.0,
            self._clock() - last_started_at,
        )

        remaining_seconds = max(
            0.0,
            self.minimum_request_interval_seconds
            - elapsed_since_previous_start,
        )

        if remaining_seconds <= 0:
            return

        self.wait_tracker.wait(
            DELIBERATE_DELAY,
            remaining_seconds,
            reason="minimum_request_interval",
            metadata={
                "label": label,
                "attempt_number": attempt_number,
                "minimum_request_interval_seconds": (
                    self.minimum_request_interval_seconds
                ),
            },
        )

    def _backoff_seconds(
        self,
        *,
        error: BaseException,
        retry_number: int,
    ) -> tuple[float, str]:
        retry_after = extract_retry_after_seconds(error)

        if retry_after is not None and retry_after > 0:
            return (
                min(
                    retry_after,
                    self.retry_policy.maximum_delay_seconds,
                ),
                "provider_retry_hint",
            )

        exponential_delay = (
            self.retry_policy.base_delay_seconds
            * (2 ** max(0, retry_number - 1))
        )

        return (
            min(
                exponential_delay,
                self.retry_policy.maximum_delay_seconds,
            ),
            "exponential_backoff",
        )

    def _should_retry(
        self,
        *,
        error: BaseException,
        completed_retry_count: int,
    ) -> tuple[bool, str]:
        if completed_retry_count >= self.retry_policy.max_retries:
            return False, "maximum_retries_reached"

        status_code = extract_status_code(error)

        if status_code not in self.retry_policy.retryable_status_codes:
            return False, "non_retryable_status"

        if status_code == 429 and is_daily_quota_error(error):
            return False, "daily_quota_exhausted"

        return True, "transient_error"

    def execute(
        self,
        operation: Callable[[], T],
        *,
        label: str = "api_request",
    ) -> RequestOutcome[T]:
        attempts: list[dict[str, Any]] = []
        wait_start_index = self.wait_tracker.event_count
        final_error: Exception | None = None
        final_value: T | None = None
        success = False

        while True:
            attempt_number = len(attempts) + 1

            self._apply_request_spacing(
                label=label,
                attempt_number=attempt_number,
            )

            attempt_started_at = self._clock()

            with self._lock:
                self._last_request_started_at = attempt_started_at

            try:
                final_value = operation()
                attempt_finished_at = self._clock()

                attempts.append(
                    {
                        "attempt_number": attempt_number,
                        "success": True,
                        "status_code": None,
                        "wall_seconds": max(
                            0.0,
                            attempt_finished_at
                            - attempt_started_at,
                        ),
                        "retry_planned": False,
                        "retry_reason": None,
                        "backoff_seconds": 0.0,
                        "error_type": None,
                        "error_message": None,
                    }
                )

                success = True
                final_error = None
                break

            except Exception as error:
                attempt_finished_at = self._clock()
                final_error = error

                completed_retry_count = len(attempts)

                should_retry, retry_reason = self._should_retry(
                    error=error,
                    completed_retry_count=completed_retry_count,
                )

                backoff_seconds = 0.0
                backoff_source: str | None = None

                if should_retry:
                    backoff_seconds, backoff_source = (
                        self._backoff_seconds(
                            error=error,
                            retry_number=(
                                completed_retry_count + 1
                            ),
                        )
                    )

                attempts.append(
                    {
                        "attempt_number": attempt_number,
                        "success": False,
                        "status_code": extract_status_code(error),
                        "wall_seconds": max(
                            0.0,
                            attempt_finished_at
                            - attempt_started_at,
                        ),
                        "retry_planned": should_retry,
                        "retry_reason": retry_reason,
                        "backoff_seconds": backoff_seconds,
                        "backoff_source": backoff_source,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

                if not should_retry:
                    break

                self.wait_tracker.wait(
                    RATE_LIMIT_BACKOFF,
                    backoff_seconds,
                    reason=retry_reason,
                    metadata={
                        "label": label,
                        "failed_attempt_number": attempt_number,
                        "status_code": extract_status_code(error),
                        "backoff_source": backoff_source,
                    },
                )

        waits = self.wait_tracker.snapshot(
            since_event_index=wait_start_index,
        )

        active_wall_seconds = sum(
            float(attempt["wall_seconds"])
            for attempt in attempts
        )
        controlled_wait_seconds = float(
            waits["controlled_wait_seconds"]
        )

        outcome = RequestOutcome(
            success=success,
            value=final_value,
            error=final_error,
            attempts=attempts,
            waits=waits,
            active_wall_seconds=active_wall_seconds,
            total_wall_seconds=(
                active_wall_seconds + controlled_wait_seconds
            ),
        )

        with self._lock:
            self._execution_reports.append(
                {
                    "label": label,
                    **outcome.to_dict(),
                }
            )

        return outcome

    def summary(self) -> dict[str, Any]:
        with self._lock:
            reports = [
                dict(report)
                for report in self._execution_reports
            ]

        successful_count = sum(
            1
            for report in reports
            if report["success"]
        )

        return {
            "execution_count": len(reports),
            "successful_execution_count": successful_count,
            "failed_execution_count": (
                len(reports) - successful_count
            ),
            "request_attempt_count": sum(
                int(report["attempt_count"])
                for report in reports
            ),
            "retry_count": sum(
                int(report["retry_count"])
                for report in reports
            ),
            "waits": self.wait_tracker.snapshot(),
            "active_wall_seconds": sum(
                float(report["active_wall_seconds"])
                for report in reports
            ),
            "total_wall_seconds": sum(
                float(report["total_wall_seconds"])
                for report in reports
            ),
            "executions": reports,
        }