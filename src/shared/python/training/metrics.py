"""Training metric data types.

Metrics are the values a training loop emits while running: per-epoch
loss, per-iteration reward, learning rate schedule samples, gradient
norms, validation accuracy, etc. They form the time series the
dashboard plots and the JSON the worker writes to disk.

Metrics are deliberately *flat* (one scalar per record). Multi-channel
quantities (e.g. per-joint losses) are encoded as separate metrics with
distinguishing ``tags``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping

from .errors import TrainingConfigError

__all__ = ["MetricKind", "TrainingMetric"]


class MetricKind(Enum):
    """Semantic category of a metric.

    Carries direction-of-improvement information so the dashboard can
    auto-pick a "best" record without per-metric configuration.
    """

    LOSS = "loss"
    REWARD = "reward"
    ACCURACY = "accuracy"
    SCALAR = "scalar"
    LEARNING_RATE = "learning_rate"
    GRAD_NORM = "grad_norm"

    @property
    def lower_is_better(self) -> bool:
        """``True`` when smaller values represent improvement."""

        return self in {MetricKind.LOSS, MetricKind.GRAD_NORM}

    @property
    def higher_is_better(self) -> bool:
        """``True`` when larger values represent improvement."""

        return self in {MetricKind.REWARD, MetricKind.ACCURACY}


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    """A single scalar observation from a running job.

    Attributes:
        name: Human-readable metric name (e.g. ``"val_loss"``).
        value: Scalar value. Must be finite (no NaN / inf).
        step: Monotone integer step counter — typically epoch index for
            supervised loops or env step for RL. ``>= 0``.
        timestamp: Wall-clock time the metric was observed, as a unix
            epoch float. ``>= 0``.
        kind: Semantic category. Defaults to :attr:`MetricKind.SCALAR`.
        tags: Free-form string metadata for grouping (e.g.
            ``{"split": "val"}``). Frozen view.

    Invariants (enforced in :meth:`__post_init__`):
        - ``name`` is a non-empty string.
        - ``value`` is a finite float.
        - ``step`` is a non-negative int.
        - ``timestamp`` is a non-negative float.
        - ``kind`` is a :class:`MetricKind` member.
        - ``tags`` keys and values are non-empty strings (values may be
          empty? No — both required non-empty for grep-ability).
    """

    name: str
    value: float
    step: int
    timestamp: float
    kind: MetricKind = MetricKind.SCALAR
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TrainingConfigError("metric name must be a non-empty string")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise TrainingConfigError(
                f"metric value must be a real number (got {self.value!r})"
            )
        if not math.isfinite(float(self.value)):
            raise TrainingConfigError(
                f"metric value must be finite (got {self.value!r})"
            )
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise TrainingConfigError(
                f"metric step must be a non-negative int (got {self.step!r})"
            )
        if (
            not isinstance(self.timestamp, (int, float))
            or isinstance(self.timestamp, bool)
            or self.timestamp < 0
        ):
            raise TrainingConfigError(
                f"metric timestamp must be a non-negative number (got {self.timestamp!r})"
            )
        if not isinstance(self.kind, MetricKind):
            raise TrainingConfigError(
                f"metric kind must be a MetricKind (got {self.kind!r})"
            )
        if not isinstance(self.tags, Mapping):
            raise TrainingConfigError("metric tags must be a Mapping")
        for key, value in self.tags.items():
            if not isinstance(key, str) or not key:
                raise TrainingConfigError(
                    f"metric tag keys must be non-empty strings (got {key!r})"
                )
            if not isinstance(value, str) or not value:
                raise TrainingConfigError(
                    f"metric tag values must be non-empty strings "
                    f"(got {value!r} for {key!r})"
                )
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))
