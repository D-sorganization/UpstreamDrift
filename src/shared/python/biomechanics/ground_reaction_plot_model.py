"""Plot data series for the ground-reaction plots (GCV-5, #11711).

Pure data, no GUI: the PyQt plots, ``GET /api/analysis/ground-reaction`` and
so the web charts all read the same :class:`GroundReactionPlotSeries`, built
from a GCV-1 :class:`GroundReactionSeries`.  A sample that cannot be computed
is ``None`` in the JSON form (NaN in the arrays), never zero.

Traces, per foot label and ``net`` (each ``{"x", "y", "z", "magnitude"}``
lists, unit in ``units``):

* ``<key>_force_n`` and, when a body weight is given, ``<key>_force_bw``;
* ``<key>_cop_m`` - centre of pressure, ``None`` below the CoP threshold;
* ``<key>_free_moment_nm`` - free moment at the CoP;
* ``<key>_moment_com_nm`` - moment about the whole-body centre of mass.

``load_share`` holds each foot's share of the summed vertical force, ``None``
when the feet together carry less than :data:`COP_MIN_FZ_N`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from typing import Any

import numpy as np

from src.shared.python.biomechanics.ground_reaction import (
    COP_MIN_FZ_N,
    NET_LABEL,
    GroundReactionSeries,
)
from src.shared.python.biomechanics.plot_traces import none_if_nan, vector_trace

__all__ = [
    "GroundReactionPlotSeries",
    "build_ground_reaction_plot_series",
    "plot_series_to_json",
    "unavailable_ground_reaction_plot",
]

#: (trace suffix, label, unit) per plotted quantity, in display order.
_QUANTITIES: tuple[tuple[str, str, str], ...] = (
    ("force_n", "Force", "N"),
    ("force_bw", "Force in Body Weights", "BW"),
    ("cop_m", "Centre of Pressure", "m"),
    ("free_moment_nm", "Free Moment", "N*m"),
    ("moment_com_nm", "Moment About CoM", "N*m"),
)


@dataclass(frozen=True)
class GroundReactionPlotSeries:
    """Time series ready to plot; ``None`` entries are unavailable samples."""

    time_s: tuple[float, ...]
    feet: tuple[str, ...]
    traces: Mapping[str, Mapping[str, list[float | None]]]
    load_share: Mapping[str, list[float | None]]
    available: bool
    reason: str = ""
    events: Mapping[str, float] = field(default_factory=dict)

    @property
    def labels(self) -> dict[str, str]:
        """Title-case display label per trace name."""
        return {name: _describe(name)[0] for name in self.traces}

    @property
    def units(self) -> dict[str, str]:
        """Unit string per trace name."""
        return {name: _describe(name)[1] for name in self.traces}

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping (``None`` for unavailable samples, no NaN)."""
        return {
            "available": self.available,
            "reason": self.reason or None,
            "time_s": list(self.time_s),
            "feet": list(self.feet),
            "events": dict(self.events),
            "units": self.units,
            "labels": self.labels,
            "load_share": {k: list(v) for k, v in self.load_share.items()},
            "traces": {
                k: {a: list(v) for a, v in t.items()} for k, t in self.traces.items()
            },
        }


def _describe(name: str) -> tuple[str, str]:
    for suffix, label, unit in _QUANTITIES:
        if name.endswith(f"_{suffix}"):
            key = name[: -len(suffix) - 1]
            return f"{key.replace('_', ' ').title()} {label}", unit
    raise ValueError(f"unknown ground-reaction trace {name!r}")


def _load_share(series: GroundReactionSeries, feet: tuple[str, ...]):
    fz = {k: series.force_n[k][:, 2] for k in feet}
    total = np.sum([fz[k] for k in feet], axis=0) if feet else np.zeros(0)
    supported = total >= COP_MIN_FZ_N
    safe = np.where(supported, total, 1.0)
    return {k: none_if_nan(np.where(supported, fz[k] / safe, np.nan)) for k in feet}


def _validated_events(events: Mapping[str, float] | None) -> dict[str, float]:
    ev = {str(k): float(v) for k, v in (events or {}).items()}
    if not all(math.isfinite(v) for v in ev.values()):
        raise ValueError("event times must be finite")
    return ev


def build_ground_reaction_plot_series(
    series: GroundReactionSeries,
    *,
    body_weight_n: float | None = None,
    events: Mapping[str, float] | None = None,
) -> GroundReactionPlotSeries:
    """Turn a GCV-1 series into plot traces.

    Args:
        series: the stacked ground-reaction breakdowns of one run.
        body_weight_n: subject weight; adds ``<key>_force_bw`` traces.
        events: named event times (address, top, impact, finish) drawn as
            markers.

    Postconditions: every trace and load-share list has one entry per sample;
    ``available`` is False only when the net reaction is never in contact.

    Raises:
        TypeError: if ``series`` is not a :class:`GroundReactionSeries`.
        ValueError: on a non-positive or non-finite body weight or a
            non-finite event time.
    """
    if not isinstance(series, GroundReactionSeries):
        raise TypeError(f"series must be a GroundReactionSeries, got {type(series)}")
    if body_weight_n is not None and not (
        math.isfinite(body_weight_n) and body_weight_n > 0.0
    ):
        raise ValueError(f"body_weight_n must be positive and finite: {body_weight_n}")
    ev = _validated_events(events)
    feet = tuple(k for k in series.force_n if k != NET_LABEL)
    tables = {
        "force_n": series.force_n,
        "cop_m": series.cop_m,
        "free_moment_nm": series.free_moment_nm,
        "moment_com_nm": series.moment_about_com_nm,
    }
    traces: dict[str, dict[str, list[float | None]]] = {}
    for key in (*feet, NET_LABEL):
        for suffix, table in tables.items():
            traces[f"{key}_{suffix}"] = vector_trace(table[key])
        if body_weight_n is not None:
            traces[f"{key}_force_bw"] = vector_trace(
                series.force_n[key] / body_weight_n
            )
    in_contact = bool(np.any(series.in_contact[NET_LABEL]))
    return GroundReactionPlotSeries(
        time_s=tuple(float(t) for t in series.times_s),
        feet=feet,
        traces=traces,
        load_share=_load_share(series, feet),
        available=in_contact,
        reason="" if in_contact else "no ground contact in this run",
        events=ev,
    )


def unavailable_ground_reaction_plot(reason: str) -> GroundReactionPlotSeries:
    """An empty series for a run or engine that cannot report ground reaction.

    Raises:
        ValueError: if ``reason`` is blank (an unavailable plot must say why).
    """
    if not str(reason).strip():
        raise ValueError("an unavailable ground-reaction plot needs a reason")
    return GroundReactionPlotSeries(
        time_s=(), feet=(), traces={}, load_share={}, available=False, reason=reason
    )


def plot_series_to_json(series: GroundReactionPlotSeries) -> str:
    """Strict JSON text (``allow_nan=False``) of ``series``."""
    return json.dumps(series.to_dict(), allow_nan=False)
