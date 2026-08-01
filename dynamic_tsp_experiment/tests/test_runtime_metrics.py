from __future__ import annotations

import time

import pytest

from src.runtime_metrics import (
    PhaseTimer,
    ResourceSampler,
)


class FakeProbe:
    def __init__(self) -> None:
        self.sample_index = 0
        self.closed = False

    def system_info(self) -> dict:
        return {
            "platform": {
                "system": "TestOS",
            },
            "cpu": {
                "logical_core_count": 4,
            },
            "memory": {
                "total_mb": 8192.0,
            },
            "local_gpu": {
                "available": False,
                "backend": "nvidia_nvml",
                "unavailable_reason": "Test ortamında GPU yok.",
                "devices": [],
            },
        }

    def sample(self) -> dict:
        self.sample_index += 1

        return {
            "system_cpu_percent": 10.0 + self.sample_index,
            "process_cpu_percent": 20.0 + self.sample_index,
            "process_memory_rss_mb": 100.0 + self.sample_index,
            "process_memory_percent": 2.0,
            "system_memory_percent": 50.0,
            "system_memory_available_mb": 4096.0,
            "local_gpu_available": False,
            "local_gpu_utilization_percent": None,
            "local_gpu_memory_used_mb": None,
            "local_gpu_memory_percent": None,
            "local_gpu_temperature_celsius": None,
        }

    def close(self) -> None:
        self.closed = True


class FailingProbe:
    def __init__(self) -> None:
        self.closed = False

    def system_info(self) -> dict:
        return {
            "available": True,
        }

    def sample(self) -> dict:
        raise RuntimeError("Örnekleme başarısız.")

    def close(self) -> None:
        self.closed = True


def test_resource_sampler_collects_and_summarizes_samples() -> None:
    probe = FakeProbe()

    sampler = ResourceSampler(
        interval_seconds=10.0,
        probe=probe,
    )

    sampler.start()
    sampler.sample_now()
    summary = sampler.stop()

    assert summary["enabled"] is True
    assert summary["sample_count"] >= 3
    assert probe.closed is True

    process_cpu = summary["overall"]["metrics"][
        "process_cpu_percent"
    ]

    assert process_cpu["sample_count"] >= 3
    assert process_cpu["minimum"] is not None
    assert process_cpu["maximum"] is not None
    assert process_cpu["average"] is not None

    assert summary["system"]["platform"]["system"] == "TestOS"
    assert summary["system"]["local_gpu"]["available"] is False


def test_resource_sampler_groups_samples_by_phase() -> None:
    probe = FakeProbe()

    sampler = ResourceSampler(
        interval_seconds=10.0,
        probe=probe,
    )

    sampler.start()

    with sampler.phase("api_call"):
        sampler.sample_now()

    with sampler.phase("route_rendering"):
        sampler.sample_now()

    summary = sampler.stop()

    assert "api_call" in summary["by_phase"]
    assert "route_rendering" in summary["by_phase"]

    assert (
        summary["by_phase"]["api_call"]["sample_count"]
        >= 1
    )
    assert (
        summary["by_phase"]["route_rendering"]["sample_count"]
        >= 1
    )


def test_resource_sampler_does_not_crash_on_sampling_error() -> None:
    probe = FailingProbe()

    sampler = ResourceSampler(
        interval_seconds=10.0,
        probe=probe,
    )

    sampler.start()
    sampler.sample_now()
    summary = sampler.stop()

    assert summary["sample_count"] == 0
    assert len(summary["sampling_errors"]) >= 1
    assert summary["sampling_errors"][0]["type"] == "RuntimeError"
    assert probe.closed is True


def test_resource_sampler_context_manager_stops_probe() -> None:
    probe = FakeProbe()

    with ResourceSampler(
        interval_seconds=0.01,
        probe=probe,
    ) as sampler:
        time.sleep(0.03)

    assert sampler.running is False
    assert probe.closed is True

    summary = sampler.summary()
    assert summary["sample_count"] >= 2


def test_resource_sampler_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError):
        ResourceSampler(
            interval_seconds=0,
            probe=FakeProbe(),
        )


def test_phase_timer_accumulates_repeated_phases() -> None:
    values = iter(
        [
            10.0,
            12.5,
            20.0,
            21.5,
        ]
    )

    timer = PhaseTimer(
        clock=lambda: next(values),
    )

    with timer.measure("api_call"):
        pass

    with timer.measure("api_call"):
        pass

    assert timer.totals()["api_call"] == pytest.approx(4.0)
    assert timer.total_seconds() == pytest.approx(4.0)


def test_phase_timer_rejects_negative_duration() -> None:
    timer = PhaseTimer()

    with pytest.raises(ValueError):
        timer.add("api_call", -1.0)