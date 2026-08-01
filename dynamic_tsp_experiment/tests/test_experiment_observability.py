from __future__ import annotations

import argparse

import pytest

from src.experiment_observability import (
    ExperimentObservability,
    ObservabilitySettings,
    add_observability_arguments,
    settings_from_args,
)


def parser_with_observability(
    *,
    include_early_stop: bool,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_observability_arguments(
        parser,
        include_early_stop=include_early_stop,
    )

    return parser


def test_default_observability_arguments() -> None:
    parser = parser_with_observability(
        include_early_stop=True
    )

    args = parser.parse_args([])

    settings = settings_from_args(
        args,
        include_early_stop=True,
    )

    assert settings.profile_resources is True
    assert settings.resource_sample_interval_seconds == 0.5
    assert settings.minimum_request_interval_seconds == 0.0
    assert settings.max_retries == 2
    assert settings.retry_base_delay_seconds == 2.0
    assert settings.retry_maximum_delay_seconds == 60.0
    assert settings.early_stop_enabled is True
    assert settings.early_stop_gap_percent == 1.0


def test_custom_observability_arguments() -> None:
    parser = parser_with_observability(
        include_early_stop=True
    )

    args = parser.parse_args(
        [
            "--no-profile-resources",
            "--resource-sample-interval-seconds",
            "1.25",
            "--request-interval-seconds",
            "20",
            "--max-retries",
            "4",
            "--retry-base-delay-seconds",
            "3",
            "--retry-maximum-delay-seconds",
            "90",
            "--early-stop-gap-percent",
            "0.5",
            "--disable-early-stop",
        ]
    )

    settings = settings_from_args(
        args,
        include_early_stop=True,
    )

    assert settings.profile_resources is False
    assert settings.resource_sample_interval_seconds == 1.25
    assert settings.minimum_request_interval_seconds == 20.0
    assert settings.max_retries == 4
    assert settings.retry_base_delay_seconds == 3.0
    assert settings.retry_maximum_delay_seconds == 90.0
    assert settings.early_stop_enabled is False
    assert settings.early_stop_gap_percent == 0.5


def test_non_iterative_runner_disables_early_stop() -> None:
    parser = parser_with_observability(
        include_early_stop=False
    )

    args = parser.parse_args([])

    settings = settings_from_args(
        args,
        include_early_stop=False,
    )

    assert settings.early_stop_enabled is False
    assert settings.early_stop_gap_percent == 1.0


def test_invalid_observability_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        ObservabilitySettings(
            resource_sample_interval_seconds=0.0
        )

    with pytest.raises(ValueError):
        ObservabilitySettings(
            minimum_request_interval_seconds=-1.0
        )

    with pytest.raises(ValueError):
        ObservabilitySettings(
            max_retries=-1
        )

    with pytest.raises(ValueError):
        ObservabilitySettings(
            retry_base_delay_seconds=10.0,
            retry_maximum_delay_seconds=5.0,
        )


def test_disabled_resource_profiling_is_explicit() -> None:
    settings = ObservabilitySettings(
        profile_resources=False,
    )

    observability = ExperimentObservability(settings)
    observability.start()
    result = observability.stop()

    assert result["resources"] == {
        "enabled": False,
        "reason": "disabled_by_configuration",
        "sample_count": 0,
    }


def test_early_stop_policy_uses_cli_settings() -> None:
    settings = ObservabilitySettings(
        profile_resources=False,
        early_stop_enabled=True,
        early_stop_gap_percent=0.75,
    )

    observability = ExperimentObservability(settings)

    policy = observability.early_stop_policy()

    assert policy.enabled is True
    assert policy.threshold_percent == 0.75
    assert policy.allowed_providers == (
        "gemini",
        "groq",
    )


def test_observability_summary_contains_separate_wait_types() -> None:
    settings = ObservabilitySettings(
        profile_resources=False,
    )

    observability = ExperimentObservability(settings)
    observability.start()
    result = observability.stop()

    waits = result["controlled_waits"]

    assert waits["deliberate_delay_seconds"] == 0
    assert waits["rate_limit_backoff_seconds"] == 0
    assert waits["controlled_wait_seconds"] == 0
