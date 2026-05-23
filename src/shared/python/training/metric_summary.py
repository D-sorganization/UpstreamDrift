"""Aggregations over :class:`TrainingMetric` streams.

The dashboard plots the raw time series, but the "Summary" panel
shows derived values: best-so-far per metric, rolling means (handy
for noisy RL episode returns), per-tag aggregates. This module owns
those helpers as pure functions / small classes so they can be unit-
tested without the dashboard.

The functions accept any iterable of :class:`TrainingMetric` rather
than coupling to a sink implementation; callers pass
``InMemoryProgressSink.metrics`` (a tuple snapshot), a JSONL-parsed
list, or whatever they have.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from .metrics import MetricKind, TrainingMetric

__all__ = [
    "BestMetric",
    "RollingMean",
    "best_per_metric",
    "filter_by_tags",
    "summarize_by_kind",
]


@dataclass(frozen=True, slots=True)
class BestMetric:
    """Best observation for a single metric name.

    "Best" follows :attr:`MetricKind.lower_is_better` /
    :attr:`MetricKind.higher_is_better`. For neutral kinds
    (``SCALAR``, ``LEARNING_RATE``) the *latest* observation is
    returned — the dashboard shows what's current, not what's optimal.
    """

    metric: TrainingMetric

    @property
    def name(self) -> str:
        return self.metric.name

    @property
    def value(self) -> float:
        return self.metric.value

    @property
    def step(self) -> int:
        return self.metric.step


def best_per_metric(
    metrics: Iterable[TrainingMetric],
) -> dict[str, BestMetric]:
    """Group ``metrics`` by name and pick the best observation per name.

    Returns:
        Mapping from metric name to :class:`BestMetric`. Empty when
        ``metrics`` is empty.
    """

    best: dict[str, TrainingMetric] = {}
    for m in metrics:
        existing = best.get(m.name)
        if existing is None:
            best[m.name] = m
            continue
        if m.kind.lower_is_better:
            if m.value < existing.value:
                best[m.name] = m
        elif m.kind.higher_is_better:
            if m.value > existing.value:
                best[m.name] = m
        else:
            # Neutral: latest by step / timestamp wins.
            if (m.step, m.timestamp) >= (existing.step, existing.timestamp):
                best[m.name] = m
    return {name: BestMetric(metric) for name, metric in best.items()}


def summarize_by_kind(
    metrics: Iterable[TrainingMetric],
) -> dict[MetricKind, tuple[TrainingMetric, ...]]:
    """Group ``metrics`` by :class:`MetricKind`.

    Useful for the dashboard: losses on one panel, rewards on another,
    learning-rate schedule on a third.
    """

    grouped: dict[MetricKind, list[TrainingMetric]] = {kind: [] for kind in MetricKind}
    for m in metrics:
        grouped[m.kind].append(m)
    return {kind: tuple(values) for kind, values in grouped.items() if values}


def filter_by_tags(
    metrics: Iterable[TrainingMetric],
    *,
    required: dict[str, str] | None = None,
    forbidden: dict[str, str] | None = None,
) -> tuple[TrainingMetric, ...]:
    """Return only metrics whose tags match the constraints.

    Args:
        metrics: Iterable of metrics to filter.
        required: Each ``(key, value)`` must appear in the metric's
            tags for the metric to pass.
        forbidden: Each ``(key, value)`` must NOT appear; if any does,
            the metric is dropped.

    Both filters are AND-combined.
    """

    req = required or {}
    forb = forbidden or {}
    out: list[TrainingMetric] = []
    for m in metrics:
        if any(m.tags.get(k) != v for k, v in req.items()):
            continue
        if any(m.tags.get(k) == v for k, v in forb.items()):
            continue
        out.append(m)
    return tuple(out)


class RollingMean:
    """Bounded-window running mean.

    Suited to RL episode-return plots where raw values are noisy and
    the dashboard wants a smoothed line. Add observations with
    :meth:`push`; read :attr:`value` for the current mean (or ``None``
    when the window is empty).
    """

    __slots__ = ("_buffer", "_sum", "_window")

    def __init__(self, window: int) -> None:
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int (got {window!r})")
        self._window = window
        self._buffer: deque[float] = deque(maxlen=window)
        self._sum: float = 0.0

    @property
    def window(self) -> int:
        return self._window

    def push(self, value: float) -> None:
        """Append ``value`` to the window. Oldest value is dropped if full."""

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"value must be a real number (got {value!r})")
        if len(self._buffer) == self._window:
            self._sum -= self._buffer[0]
        self._buffer.append(float(value))
        self._sum += float(value)

    @property
    def value(self) -> float | None:
        """Current mean, or ``None`` when no observations have been pushed."""

        if not self._buffer:
            return None
        return self._sum / len(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._sum = 0.0
