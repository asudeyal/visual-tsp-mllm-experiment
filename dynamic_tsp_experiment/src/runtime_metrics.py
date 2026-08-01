from __future__ import annotations

import math
import platform
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import psutil


BYTES_PER_MEBIBYTE = 1024 * 1024

RESOURCE_METRIC_KEYS = (
    "system_cpu_percent",
    "process_cpu_percent",
    "process_memory_rss_mb",
    "process_memory_percent",
    "system_memory_percent",
    "system_memory_available_mb",
    "local_gpu_utilization_percent",
    "local_gpu_memory_used_mb",
    "local_gpu_memory_percent",
    "local_gpu_temperature_celsius",
)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
        return None

    converted = float(value)

    if not math.isfinite(converted):
        return None

    return converted


def _metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "minimum": None,
            "maximum": None,
            "average": None,
            "last": None,
        }

    return {
        "sample_count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "average": sum(values) / len(values),
        "last": values[-1],
    }


def _summarize_resource_samples(
    samples: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for metric_name in RESOURCE_METRIC_KEYS:
        values = [
            converted
            for sample in samples
            if (converted := _finite_float(sample.get(metric_name)))
            is not None
        ]

        summary[metric_name] = _metric_summary(values)

    return summary


def _safe_exception_text(error: BaseException) -> str:
    message = str(error).strip()

    if message:
        return f"{type(error).__name__}: {message}"

    return type(error).__name__


def _decode_nvml_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


class NvidiaGpuProbe:
    """
    NVIDIA GPU ölçümlerini NVML üzerinden almaya çalışır.

    NVIDIA GPU, NVML kütüphanesi veya uygun sürücü bulunmaması hata
    olarak dışarı fırlatılmaz. Bunun yerine available=False döndürülür.
    """

    def __init__(self) -> None:
        self._module: Any = None
        self._handles: list[Any] = []
        self._devices: list[dict[str, Any]] = []
        self._initialized = False
        self._available = False
        self._unavailable_reason: str | None = None

        self._initialize()

    def _initialize(self) -> None:
        try:
            import pynvml
        except Exception as error:
            self._unavailable_reason = _safe_exception_text(error)
            return

        self._module = pynvml

        try:
            pynvml.nvmlInit()
            self._initialized = True

            device_count = int(pynvml.nvmlDeviceGetCount())

            if device_count <= 0:
                self._unavailable_reason = "NVIDIA GPU bulunamadı."
                self.close()
                return

            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = _decode_nvml_text(
                    pynvml.nvmlDeviceGetName(handle)
                )

                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                self._handles.append(handle)
                self._devices.append(
                    {
                        "index": index,
                        "name": name,
                        "memory_total_mb": (
                            float(memory.total) / BYTES_PER_MEBIBYTE
                        ),
                    }
                )

            self._available = bool(self._handles)

        except Exception as error:
            self._unavailable_reason = _safe_exception_text(error)
            self.close()

    @property
    def status(self) -> dict[str, Any]:
        return {
            "available": self._available,
            "backend": "nvidia_nvml",
            "unavailable_reason": self._unavailable_reason,
            "devices": [dict(device) for device in self._devices],
        }

    def sample(self) -> dict[str, Any]:
        if not self._available or self._module is None:
            return {
                "local_gpu_available": False,
            }

        utilization_values: list[float] = []
        memory_used_values: list[float] = []
        memory_total_values: list[float] = []
        temperature_values: list[float] = []

        for handle in self._handles:
            try:
                utilization = (
                    self._module.nvmlDeviceGetUtilizationRates(handle)
                )
                utilization_values.append(float(utilization.gpu))
            except Exception:
                pass

            try:
                memory = self._module.nvmlDeviceGetMemoryInfo(handle)

                memory_used_values.append(
                    float(memory.used) / BYTES_PER_MEBIBYTE
                )
                memory_total_values.append(
                    float(memory.total) / BYTES_PER_MEBIBYTE
                )
            except Exception:
                pass

            try:
                temperature = self._module.nvmlDeviceGetTemperature(
                    handle,
                    self._module.NVML_TEMPERATURE_GPU,
                )
                temperature_values.append(float(temperature))
            except Exception:
                pass

        memory_used_mb = (
            sum(memory_used_values)
            if memory_used_values
            else None
        )
        memory_total_mb = (
            sum(memory_total_values)
            if memory_total_values
            else None
        )

        memory_percent: float | None = None

        if (
            memory_used_mb is not None
            and memory_total_mb is not None
            and memory_total_mb > 0
        ):
            memory_percent = (
                memory_used_mb / memory_total_mb
            ) * 100.0

        return {
            "local_gpu_available": True,
            "local_gpu_utilization_percent": (
                max(utilization_values)
                if utilization_values
                else None
            ),
            "local_gpu_memory_used_mb": memory_used_mb,
            "local_gpu_memory_percent": memory_percent,
            "local_gpu_temperature_celsius": (
                max(temperature_values)
                if temperature_values
                else None
            ),
        }

    def close(self) -> None:
        if self._initialized and self._module is not None:
            try:
                self._module.nvmlShutdown()
            except Exception:
                pass

        self._initialized = False


class LocalResourceProbe:
    """
    Mevcut Python sürecinin ve yerel bilgisayarın kaynak kullanımını ölçer.
    """

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._gpu = NvidiaGpuProbe()
        self._closed = False

        # İlk cpu_percent çağrısı sonraki ölçüm için başlangıç oluşturur.
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    def system_info(self) -> dict[str, Any]:
        virtual_memory = psutil.virtual_memory()
        processor_name = platform.processor().strip() or None

        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "cpu": {
                "processor": processor_name,
                "physical_core_count": psutil.cpu_count(logical=False),
                "logical_core_count": psutil.cpu_count(logical=True),
            },
            "memory": {
                "total_mb": (
                    float(virtual_memory.total)
                    / BYTES_PER_MEBIBYTE
                ),
            },
            "local_gpu": self._gpu.status,
        }

    def sample(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Kaynak ölçüm nesnesi kapatılmış.")

        process_memory = self._process.memory_info()
        system_memory = psutil.virtual_memory()

        sample = {
            "system_cpu_percent": float(
                psutil.cpu_percent(interval=None)
            ),
            # Çok çekirdek kullanan bir süreçte bu değer 100'ü aşabilir.
            "process_cpu_percent": float(
                self._process.cpu_percent(interval=None)
            ),
            "process_memory_rss_mb": (
                float(process_memory.rss)
                / BYTES_PER_MEBIBYTE
            ),
            "process_memory_percent": float(
                self._process.memory_percent()
            ),
            "system_memory_percent": float(
                system_memory.percent
            ),
            "system_memory_available_mb": (
                float(system_memory.available)
                / BYTES_PER_MEBIBYTE
            ),
        }

        sample.update(self._gpu.sample())
        return sample

    def close(self) -> None:
        if self._closed:
            return

        self._gpu.close()
        self._closed = True


class ResourceSampler:
    """
    Kaynak kullanımını arka planda belirli aralıklarla örnekler.

    Örnekler çalışma geneli ve faz bazında özetlenebilir.
    """

    def __init__(
        self,
        *,
        interval_seconds: float = 0.5,
        probe: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(
                "interval_seconds sıfırdan büyük olmalıdır."
            )

        self.interval_seconds = float(interval_seconds)
        self._probe = probe if probe is not None else LocalResourceProbe()
        self._clock = clock

        self._state_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []

        self._current_phase = "unclassified"
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._running = False
        self._closed = False

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    def start(self) -> ResourceSampler:
        with self._state_lock:
            if self._running:
                return self

            if self._closed:
                raise RuntimeError(
                    "Kapatılmış ResourceSampler yeniden başlatılamaz."
                )

            self._started_at = self._clock()
            self._finished_at = None
            self._running = True
            self._stop_event.clear()

        # Çok kısa çalışmaların da en az bir örneği olsun.
        self.sample_now()

        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="resource-sampler",
            daemon=True,
        )
        self._thread.start()

        return self

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.sample_now()

    def sample_now(self) -> dict[str, Any] | None:
        with self._sample_lock:
            with self._state_lock:
                if not self._running or self._started_at is None:
                    return None

                phase = self._current_phase
                started_at = self._started_at

            elapsed = max(0.0, self._clock() - started_at)

            try:
                raw_sample = dict(self._probe.sample())
            except Exception as error:
                error_record = {
                    "elapsed_seconds": elapsed,
                    "phase": phase,
                    "type": type(error).__name__,
                    "message": str(error),
                }

                with self._state_lock:
                    self._errors.append(error_record)

                return None

            sample = {
                "elapsed_seconds": elapsed,
                "phase": phase,
                **raw_sample,
            }

            with self._state_lock:
                self._samples.append(sample)

            return dict(sample)

    def set_phase(self, phase: str) -> None:
        normalized = str(phase).strip()

        if not normalized:
            raise ValueError("Faz adı boş olamaz.")

        with self._state_lock:
            self._current_phase = normalized

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        normalized = str(phase).strip()

        if not normalized:
            raise ValueError("Faz adı boş olamaz.")

        with self._state_lock:
            previous_phase = self._current_phase
            self._current_phase = normalized

        try:
            yield
        finally:
            with self._state_lock:
                self._current_phase = previous_phase

    def stop(self) -> dict[str, Any]:
        with self._state_lock:
            if self._closed:
                return self.summary()

            if not self._running:
                self._close_probe()
                return self.summary()

        self._stop_event.set()

        thread = self._thread
        if thread is not None:
            thread.join(
                timeout=max(1.0, self.interval_seconds * 3.0)
            )

        # Son durumu da yakalamak için kapanmadan önce son örnek.
        self.sample_now()

        with self._state_lock:
            self._finished_at = self._clock()
            self._running = False

        self._close_probe()
        return self.summary()

    def _close_probe(self) -> None:
        with self._state_lock:
            if self._closed:
                return

            self._closed = True

        close_method = getattr(self._probe, "close", None)

        if callable(close_method):
            try:
                close_method()
            except Exception as error:
                with self._state_lock:
                    self._errors.append(
                        {
                            "elapsed_seconds": self.duration_seconds(),
                            "phase": "shutdown",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )

    def duration_seconds(self) -> float:
        with self._state_lock:
            started_at = self._started_at
            finished_at = self._finished_at
            running = self._running

        if started_at is None:
            return 0.0

        endpoint = (
            self._clock()
            if running or finished_at is None
            else finished_at
        )

        return max(0.0, endpoint - started_at)

    def summary(self) -> dict[str, Any]:
        with self._state_lock:
            samples = [dict(sample) for sample in self._samples]
            errors = [dict(error) for error in self._errors]

        samples_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(
            list
        )

        for sample in samples:
            phase = str(sample.get("phase") or "unclassified")
            samples_by_phase[phase].append(sample)

        by_phase = {
            phase: {
                "sample_count": len(phase_samples),
                "metrics": _summarize_resource_samples(
                    phase_samples
                ),
            }
            for phase, phase_samples in sorted(samples_by_phase.items())
        }

        try:
            system_info = dict(self._probe.system_info())
        except Exception as error:
            system_info = {
                "available": False,
                "error": _safe_exception_text(error),
            }

        return {
            "enabled": True,
            "sampling_interval_seconds": self.interval_seconds,
            "duration_seconds": self.duration_seconds(),
            "sample_count": len(samples),
            "system": system_info,
            "overall": {
                "sample_count": len(samples),
                "metrics": _summarize_resource_samples(samples),
            },
            "by_phase": by_phase,
            "sampling_errors": errors,
        }

    def __enter__(self) -> ResourceSampler:
        return self.start()

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.stop()


class PhaseTimer:
    """
    Aynı isimli fazların sürelerini biriktiren basit wall-clock ölçer.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._clock = clock
        self._durations: defaultdict[str, float] = defaultdict(float)

    def add(self, phase: str, seconds: float) -> None:
        normalized = str(phase).strip()

        if not normalized:
            raise ValueError("Faz adı boş olamaz.")

        converted = float(seconds)

        if converted < 0 or not math.isfinite(converted):
            raise ValueError(
                "Faz süresi sonlu ve negatif olmayan bir sayı olmalıdır."
            )

        self._durations[normalized] += converted

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        started_at = self._clock()

        try:
            yield
        finally:
            self.add(
                phase,
                max(0.0, self._clock() - started_at),
            )

    def totals(self) -> dict[str, float]:
        return dict(sorted(self._durations.items()))

    def total_seconds(self) -> float:
        return sum(self._durations.values())