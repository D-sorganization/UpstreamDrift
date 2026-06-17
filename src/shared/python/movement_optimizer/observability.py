# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Small in-process metrics recorder for runtime observability."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class MetricSample:
    """A single point-in-time metric sample."""

    name: str
    value: float
    labels: dict[str, str]


class InMemoryMetrics:
    """Thread-safe metric sink used by CLI, GUI, and tests."""

    def __init__(self) -> None:
        self._samples: list[MetricSample] = []
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1.0, **labels: object) -> None:
        """Record a counter-style metric increment."""
        self.record(name, value, **labels)

    def observe(self, name: str, value: float, **labels: object) -> None:
        """Record a gauge/timing-style metric observation."""
        self.record(name, value, **labels)

    def record(self, name: str, value: float, **labels: object) -> None:
        """Append a metric sample with stringified labels."""
        if not name:
            raise ValueError("metric name must be non-empty")
        sample = MetricSample(
            name=name,
            value=float(value),
            labels={key: str(val) for key, val in labels.items()},
        )
        with self._lock:
            self._samples.append(sample)

    def snapshot(self) -> list[MetricSample]:
        """Return a copy of recorded samples."""
        with self._lock:
            return list(self._samples)

    def clear(self) -> None:
        """Remove all recorded samples."""
        with self._lock:
            self._samples.clear()


metrics: Final = InMemoryMetrics()
