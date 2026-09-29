"""Host-resource monitor for the training dashboard.

Samples CPU, memory, and (optionally) GPU usage at a low frequency and
exposes the most recent observation as a frozen :class:`ResourceSample`.
The dashboard subscribes for live updates; the scheduler can consult
the latest sample before admitting a job to avoid oversubscribing the
host.

Optional dependencies:
    - psutil: CPU + memory readings. Required to construct the
      monitor; absence raises :class:`ResourceMonitorUnavailableError`
      with a clear install hint.
    - pynvml: GPU utilization + memory. Optional; when missing the
      :attr:`ResourceSample.gpus` field is an empty tuple.

The monitor itself runs a daemon thread; callers must call
:meth:`ResourceMonitor.stop` (typically on launcher shutdown).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .errors import TrainingError

__all__ = [
    "GpuSample",
    "ResourceMonitor",
    "ResourceMonitorUnavailableError",
    "ResourceSample",
]


logger = logging.getLogger(__name__)


class ResourceMonitorUnavailableError(TrainingError, ImportError):
    """Raised when psutil isn't installed and the monitor cannot start.

    The training subsystem treats resource monitoring as optional —
    consumers must catch this and degrade their UI accordingly.
    """


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One sample of a single GPU's utilization.

    Attributes:
        index: 0-based device index.
        name: Vendor-reported device name (e.g. ``"NVIDIA RTX A6000"``).
        utilization_percent: Compute utilization in [0.0, 100.0], or
            ``None`` when the driver does not report it.
        memory_used_mb: Memory currently in use, in MiB.
        memory_total_mb: Total device memory, in MiB.
    """

    index: int
    name: str
    utilization_percent: float | None
    memory_used_mb: int
    memory_total_mb: int

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"index must be a non-negative int (got {self.index!r})")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.utilization_percent is not None:
            if not isinstance(self.utilization_percent, (int, float)):
                raise ValueError(
                    f"utilization_percent must be a number or None "
                    f"(got {self.utilization_percent!r})"
                )
            if not 0.0 <= float(self.utilization_percent) <= 100.0:
                raise ValueError(
                    f"utilization_percent must be in [0, 100] "
                    f"(got {self.utilization_percent!r})"
                )
        if self.memory_used_mb < 0 or self.memory_total_mb < 0:
            raise ValueError("memory values must be non-negative")
        if self.memory_used_mb > self.memory_total_mb:
            raise ValueError(
                f"memory_used_mb ({self.memory_used_mb}) cannot exceed "
                f"memory_total_mb ({self.memory_total_mb})"
            )


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One snapshot of host resource usage.

    Attributes:
        timestamp: Unix epoch seconds the sample was captured.
        cpu_percent: Overall CPU utilization in [0.0, 100.0].
        memory_used_mb: Resident memory currently in use, in MiB.
        memory_total_mb: Total host memory, in MiB.
        gpus: Per-GPU samples. Empty tuple when no GPU library is
            available.
    """

    timestamp: float
    cpu_percent: float
    memory_used_mb: int
    memory_total_mb: int
    gpus: tuple[GpuSample, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if not 0.0 <= float(self.cpu_percent) <= 100.0:
            raise ValueError("cpu_percent must be in [0, 100]")
        if self.memory_used_mb < 0 or self.memory_total_mb <= 0:
            raise ValueError("memory values must be positive")
        if not isinstance(self.gpus, tuple):
            raise TypeError("gpus must be a tuple")

    @property
    def memory_percent(self) -> float:
        """Derived: memory utilization as a percentage."""

        return 100.0 * self.memory_used_mb / self.memory_total_mb


class ResourceMonitor:
    """Background sampler exposing the most recent :class:`ResourceSample`.

    Args:
        sample_interval_s: Seconds between samples. Defaults to 1.0;
            anything faster than 0.25 raises :class:`ValueError` to
            keep the monitor from oversampling.
        on_sample: Optional callback invoked from the sampler thread
            after each sample. Tests use it as a fast notification
            channel; the dashboard uses it to mark its plot dirty.
        clock: Time source for the sample timestamp. Defaulted to
            :func:`time.time` so tests can inject a deterministic clock.
    """

    __slots__ = (
        "_clock",
        "_gpu_handles",
        "_interval",
        "_last",
        "_lock",
        "_on_sample",
        "_psutil",
        "_stop_event",
        "_thread",
    )

    def __init__(
        self,
        *,
        sample_interval_s: float = 1.0,
        on_sample: Callable[[ResourceSample], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(sample_interval_s, (int, float)) or sample_interval_s < 0.25:
            raise ValueError(
                f"sample_interval_s must be >= 0.25 (got {sample_interval_s!r})"
            )
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable or None")
        try:
            import psutil  # noqa: PLC0415 - optional dep import
        except ImportError as exc:  # pragma: no cover - exercised via skipif
            raise ResourceMonitorUnavailableError(
                "psutil is required for resource monitoring. "
                "Install it via `pip install psutil`."
            ) from exc
        self._psutil = psutil
        self._gpu_handles = _try_init_nvml()
        self._interval = float(sample_interval_s)
        self._on_sample = on_sample
        self._clock = clock
        self._last: ResourceSample | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling in a daemon thread. Idempotent."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="training-resource-monitor",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, wait: bool = True) -> None:
        """Signal the sampler thread to stop and optionally join it."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None and wait:
            thread.join(timeout=2.0)

    @property
    def latest(self) -> ResourceSample | None:
        """Most recent sample, or ``None`` if the monitor never sampled."""

        with self._lock:
            return self._last

    def sample_once(self) -> ResourceSample:
        """Take a single sample synchronously (used by tests and admission)."""

        cpu = float(self._psutil.cpu_percent(interval=None))
        vm = self._psutil.virtual_memory()
        sample = ResourceSample(
            timestamp=self._clock(),
            cpu_percent=cpu,
            memory_used_mb=int(vm.used // (1024 * 1024)),
            memory_total_mb=int(vm.total // (1024 * 1024)),
            gpus=_sample_gpus(self._gpu_handles),
        )
        with self._lock:
            self._last = sample
        if self._on_sample is not None:
            try:
                self._on_sample(sample)
            except (RuntimeError, ValueError, TypeError, OSError):
                logger.exception("resource-monitor on_sample callback raised")
        return sample

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sample_once()
            except (RuntimeError, ValueError, OSError):
                logger.exception("resource-monitor sample failed; continuing")
            self._stop_event.wait(timeout=self._interval)


def _try_init_nvml() -> tuple[object, ...]:
    """Attempt to enumerate NVIDIA GPU handles via pynvml.

    Returns an empty tuple when pynvml isn't installed or no devices
    are present — callers don't need to special-case the absence.
    """

    try:
        import pynvml  # noqa: PLC0415 - optional dep import
    except ImportError:
        return ()
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        return tuple(pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count))
    except (RuntimeError, OSError):
        logger.exception("pynvml init failed; GPU samples unavailable")
        return ()


def _sample_gpus(handles: tuple[object, ...]) -> tuple[GpuSample, ...]:
    if not handles:
        return ()
    try:
        import pynvml  # noqa: PLC0415 - optional dep import
    except ImportError:  # pragma: no cover - handles were enumerated via pynvml
        return ()
    samples: list[GpuSample] = []
    for index, handle in enumerate(handles):
        try:
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = (
                name_raw.decode("utf-8")
                if isinstance(name_raw, bytes)
                else str(name_raw)
            )
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            samples.append(
                GpuSample(
                    index=index,
                    name=name or f"gpu-{index}",
                    utilization_percent=float(util.gpu),
                    memory_used_mb=int(mem.used // (1024 * 1024)),
                    memory_total_mb=int(mem.total // (1024 * 1024)),
                )
            )
        except (RuntimeError, ValueError, OSError):
            logger.exception("failed to sample GPU %d", index)
    return tuple(samples)
