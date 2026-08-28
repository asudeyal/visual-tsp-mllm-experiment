from __future__ import annotations

from src.experiment.compact import trace_api_metrics, trace_provider_wait_seconds
from src.providers.base import ProviderAdapter
from src.schemas import ProviderCapabilities, ProviderResponse


class _FakeResponse:
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"{status_code} synthetic")
        self.response = _FakeResponse(status_code, retry_after)


class _RetryProvider(ProviderAdapter):
    provider_id = "retry-test"
    capabilities = ProviderCapabilities()

    def __init__(self, *, retry_after: str | None) -> None:
        super().__init__("retry-model", request_retries=2)
        self.retry_after = retry_after
        self.calls = 0

    def _generate_once(self, parts, *, phase, temperature, thinking_level):
        self.calls += 1
        if self.calls == 1:
            raise _FakeHTTPError(429, self.retry_after)
        return ProviderResponse(
            text="ok",
            provider=self.provider_id,
            model=self.model,
            phase=phase,
            usage={"total_tokens": 12},
        )


def test_retry_after_and_request_delay_are_both_respected(monkeypatch) -> None:
    import src.providers.base as base_module

    clock = {"now": 0.0}
    sleeps: list[float] = []

    monkeypatch.setattr(base_module.time, "monotonic", lambda: clock["now"])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(float(seconds))
        clock["now"] += float(seconds)

    monkeypatch.setattr(base_module.time, "sleep", fake_sleep)

    provider = _RetryProvider(retry_after="7")
    provider.configure_request_delay(10)
    response = provider.generate([], phase="test")

    assert provider.calls == 2
    assert sleeps == [7.0, 3.0]
    assert response.raw_metadata["retry_backoff_wait_seconds"] == 7.0
    assert response.raw_metadata["request_delay_wait_seconds"] == 3.0
    assert response.raw_metadata["provider_wait_seconds"] == 10.0


def test_429_without_retry_after_uses_30_second_first_backoff(monkeypatch) -> None:
    import src.providers.base as base_module

    sleeps: list[float] = []
    monkeypatch.setattr(base_module.time, "sleep", lambda seconds: sleeps.append(float(seconds)))

    provider = _RetryProvider(retry_after=None)
    response = provider.generate([], phase="test")

    assert provider.calls == 2
    assert sleeps == [30.0]
    assert response.raw_metadata["retry_backoff_wait_seconds"] == 30.0


def test_non_transient_http_error_is_not_retried(monkeypatch) -> None:
    class BadRequestProvider(_RetryProvider):
        def _generate_once(self, parts, *, phase, temperature, thinking_level):
            self.calls += 1
            raise _FakeHTTPError(401)

    import src.providers.base as base_module

    sleeps: list[float] = []
    monkeypatch.setattr(base_module.time, "sleep", lambda seconds: sleeps.append(float(seconds)))
    provider = BadRequestProvider(retry_after=None)

    try:
        provider.generate([], phase="test")
    except _FakeHTTPError as exc:
        assert exc.response.status_code == 401
    else:
        raise AssertionError("401 retry edilmemeliydi")

    assert provider.calls == 1
    assert sleeps == []


def test_compact_metrics_support_openai_compatible_total_tokens_and_waits() -> None:
    events = [
        {
            "event": "agent_call",
            "call": {
                "usage": {"total_tokens": 123},
                "latency_seconds": 1.5,
                "raw_metadata": {"provider_wait_seconds": 10.0},
            },
        },
        {
            "event": "provider_error",
            "provider_wait_seconds": 30.0,
        },
    ]

    api_calls, tokens, active, errors = trace_api_metrics(events)
    assert api_calls == 1
    assert tokens == 123
    assert active == 1.5
    assert errors == 1
    assert trace_provider_wait_seconds(events) == 40.0


def test_read_timeout_is_not_retried(monkeypatch) -> None:
    from requests.exceptions import ReadTimeout
    import src.providers.base as base_module

    class TimeoutProvider(_RetryProvider):
        def _generate_once(self, parts, *, phase, temperature, thinking_level):
            self.calls += 1
            raise ReadTimeout("synthetic read timeout")

    sleeps: list[float] = []
    monkeypatch.setattr(
        base_module.time,
        "sleep",
        lambda seconds: sleeps.append(float(seconds)),
    )
    provider = TimeoutProvider(retry_after=None)

    try:
        provider.generate([], phase="test")
    except ReadTimeout:
        pass
    else:
        raise AssertionError("ReadTimeout doğrudan yüzeye çıkmalıydı")

    assert provider.calls == 1
    assert sleeps == []
