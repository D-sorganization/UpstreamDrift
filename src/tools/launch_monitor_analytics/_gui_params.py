"""Frozen parameter/data objects for the Launch Monitor Analytics GUI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ELLIPSE_POINTS = 200


@dataclass(frozen=True, slots=True)
class _RelationshipParams:
    """Widget-derived inputs for one relationship analysis (GUI-thread read)."""

    metrics: tuple[str, ...]
    controls: tuple[str, ...]
    method: str
    edge_threshold: float


@dataclass(frozen=True, slots=True)
class _ModelParams:
    """Widget-derived inputs for one predictive-model fit (GUI-thread read)."""

    target: str
    features: tuple[str, ...]
    model: str
    random_seed: int
    group_column: str | None


@dataclass(frozen=True, slots=True)
class _ComparisonParams:
    """Widget-derived inputs for one monitor comparison (GUI-thread read)."""

    metric: str
    match_column: str | None
    reference_monitor: str | None


@dataclass(frozen=True, slots=True)
class _DispersionParams:
    """Widget-derived inputs for one dispersion analysis (GUI-thread read)."""

    forward: str
    lateral: str
    group_column: str


@dataclass(frozen=True, slots=True)
class _TrendParams:
    """Widget-derived inputs for one longitudinal trend (GUI-thread read)."""

    metric: str
    time_column: str
    rolling_window: int


@dataclass(frozen=True, slots=True)
class _DispersionPlotData:
    """Plain per-group plot geometry for one dispersion group (worker-safe)."""

    name: str
    forward_values: np.ndarray
    lateral_values: np.ndarray
    ellipse_forward: np.ndarray
    ellipse_lateral: np.ndarray


@dataclass(frozen=True, slots=True)
class _DispersionData:
    """Table rows plus plot geometry for a completed dispersion analysis."""

    rows: list[dict[str, object]]
    series: tuple[_DispersionPlotData, ...]
