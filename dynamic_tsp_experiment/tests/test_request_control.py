from __future__ import annotations

import pytest

from src.request_control import (
    DELIBERATE_DELAY,
    RATE_LIMIT_BACKOFF,
    RequestController,
    RetryPolicy,
    WaitTracker,
    extract_provider_timing,
    extract_retry_after_seconds,
    extract_status_code,
    is_daily_quota_error,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


class FakeApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


def make_controller(
    clock: FakeClock,
    *,
    max_retries: int = 2,
    minimum_interval: float = 0.0,
    base_delay: float = 2.0,
) -> RequestController:
    tracker = WaitTracker(
        sleeper=clock.sleep,
        clock=clock,
    )

    return RequestController(
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            base_delay_seconds=base_delay,
            maximum_delay_seconds=60.0,
        ),
        minimum_request_interval_seconds=minimum_interval,
        wait_tracker=tracker,
        clock=clock,
    )


def test_extract_status_code_from_attributes_and_messages() -> None:
    attribute_error = FakeApiError(
        "Rate limited",
        status_code=429,
    )

    assert extract_status_code(attribute_error) == 429

    message_error = RuntimeError(
        'OpenRouter HTTP 503: {"error": "unavailable"}'
    )

    assert extract_status_code(message_error) == 503

    gemini_error = RuntimeError(
        "429 RESOURCE_EXHAUSTED"
    )

    assert extract_status_code(gemini_error) == 429


def test_extract_retry_after_from_header_and_messages() -> None:
    header_error = FakeApiError(
        "Rate limited",
        status_code=429,
        headers={
            "Retry-After": "4.5",
        },
    )

    assert extract_retry_after_seconds(
        header_error
    ) == pytest.approx(4.5)

    seconds_error = RuntimeError(
        "Please retry in 17.728601887s."
    )

    assert extract_retry_after_seconds(
        seconds_error
    ) == pytest.approx(17.728601887)

    milliseconds_error = RuntimeError(
        "'retryDelay': '82.441033ms'"
    )

    assert extract_retry_after_seconds(
        milliseconds_error
    ) == pytest.approx(0.082441033)


def test_daily_quota_is_detected() -> None:
    error = FakeApiError(
        "QuotaId: GenerateRequestsPerDayPerProjectPerModel",
        status_code=429,
    )

    assert is_daily_quota_error(error) is True


def test_wait_tracker_separates_wait_types() -> None:
    clock = FakeClock()

    tracker = WaitTracker(
        sleeper=clock.sleep,
        clock=clock,
    )

    tracker.wait(
        DELIBERATE_DELAY,
        2.0,
        reason="minimum_request_interval",
    )
    tracker.wait(
        RATE_LIMIT_BACKOFF,
        3.5,
        reason="transient_error",
    )

    result = tracker.snapshot()

    assert result["event_count"] == 2
    assert result["deliberate_delay_seconds"] == pytest.approx(2.0)
    assert result["rate_limit_backoff_seconds"] == pytest.approx(3.5)
    assert result["controlled_wait_seconds"] == pytest.approx(5.5)


def test_request_controller_enforces_minimum_interval() -> None:
    clock = FakeClock()

    controller = make_controller(
        clock,
        minimum_interval=3.0,
    )

    first = controller.execute(
        lambda: "first",
        label="first_request",
    )

    clock.advance(1.0)

    second = controller.execute(
        lambda: "second",
        label="second_request",
    )

    assert first.unwrap() == "first"
    assert second.unwrap() == "second"

    assert second.waits[
        "deliberate_delay_seconds"
    ] == pytest.approx(2.0)

    assert second.waits[
        "rate_limit_backoff_seconds"
    ] == pytest.approx(0.0)


def test_request_controller_retries_429_with_provider_hint() -> None:
    clock = FakeClock()
    controller = make_controller(clock)

    call_count = 0

    def operation() -> str:
        nonlocal call_count
        call_count += 1
        clock.advance(0.25)

        if call_count == 1:
            raise FakeApiError(
                "Please retry in 4.25s.",
                status_code=429,
            )

        return "success"

    outcome = controller.execute(
        operation,
        label="groq_route_generation",
    )

    assert outcome.success is True
    assert outcome.unwrap() == "success"
    assert outcome.attempt_count == 2
    assert outcome.retry_count == 1

    assert outcome.waits[
        "rate_limit_backoff_seconds"
    ] == pytest.approx(4.25)

    assert outcome.active_wall_seconds == pytest.approx(0.5)
    assert outcome.total_wall_seconds == pytest.approx(4.75)

    assert outcome.attempts[0]["retry_planned"] is True
    assert (
        outcome.attempts[0]["backoff_source"]
        == "provider_retry_hint"
    )


def test_request_controller_uses_exponential_backoff() -> None:
    clock = FakeClock()

    controller = make_controller(
        clock,
        max_retries=2,
        base_delay=2.0,
    )

    call_count = 0

    def operation() -> str:
        nonlocal call_count
        call_count += 1

        if call_count <= 2:
            raise FakeApiError(
                "Service unavailable",
                status_code=503,
            )

        return "recovered"

    outcome = controller.execute(operation)

    assert outcome.unwrap() == "recovered"
    assert outcome.attempt_count == 3
    assert clock.sleeps == pytest.approx([2.0, 4.0])

    assert outcome.waits[
        "rate_limit_backoff_seconds"
    ] == pytest.approx(6.0)


def test_daily_quota_is_not_retried() -> None:
    clock = FakeClock()
    controller = make_controller(clock)

    error = FakeApiError(
        "GenerateRequestsPerDayPerProjectPerModel quota exceeded",
        status_code=429,
    )

    outcome = controller.execute(
        lambda: (_ for _ in ()).throw(error)
    )

    assert outcome.success is False
    assert outcome.attempt_count == 1
    assert outcome.retry_count == 0

    assert (
        outcome.attempts[0]["retry_reason"]
        == "daily_quota_exhausted"
    )

    assert outcome.waits[
        "controlled_wait_seconds"
    ] == pytest.approx(0.0)

    with pytest.raises(FakeApiError):
        outcome.unwrap()


def test_non_retryable_error_is_not_retried() -> None:
    clock = FakeClock()
    controller = make_controller(clock)

    outcome = controller.execute(
        lambda: (_ for _ in ()).throw(
            FakeApiError(
                "Bad request",
                status_code=400,
            )
        )
    )

    assert outcome.success is False
    assert outcome.attempt_count == 1
    assert outcome.retry_count == 0
    assert clock.sleeps == []


def test_extract_provider_timing_from_groq_usage() -> None:
    usage = {
        "prompt_token_count": 1503,
        "candidates_token_count": 80,
        "total_token_count": 1583,
        "raw": {
            "queue_time": 0.55,
            "prompt_time": 0.13,
            "completion_time": 0.16,
            "total_time": 0.84,
        },
    }

    timing = extract_provider_timing(
        usage,
        api_call_wall_seconds=1.20,
    )

    assert timing["available"] is True
    assert timing[
        "provider_queue_seconds"
    ] == pytest.approx(0.55)

    assert timing[
        "provider_prompt_seconds"
    ] == pytest.approx(0.13)

    assert timing[
        "provider_completion_seconds"
    ] == pytest.approx(0.16)

    assert timing[
        "provider_total_seconds"
    ] == pytest.approx(0.84)

    assert timing[
        "estimated_network_or_client_overhead_seconds"
    ] == pytest.approx(0.36)


def test_provider_timing_is_optional() -> None:
    timing = extract_provider_timing(
        {
            "total_token_count": 100,
        },
        api_call_wall_seconds=2.0,
    )

    assert timing["available"] is False
    assert timing["provider_queue_seconds"] is None
    assert timing["provider_total_seconds"] is None
    assert (
        timing[
            "estimated_network_or_client_overhead_seconds"
        ]
        is None
    )

def test_keyboard_interrupt_is_not_captured() -> None:
    clock = FakeClock()
    controller = make_controller(clock)

    def interrupted_operation() -> str:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        controller.execute(interrupted_operation)

    summary = controller.summary()

    assert summary["execution_count"] == 0
    assert summary["request_attempt_count"] == 0
    assert clock.sleeps == []