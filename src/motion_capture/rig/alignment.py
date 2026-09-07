"""Apply the session's strobe alignment to ingested observations — as evidence.

Ingest stamps every observation with ``time_s`` on its own camera's arrival
clock. When the session manifest carries a ``timing`` block (``capture
--timing``), this module adds, beside that value and never in place of it,
the same instant expressed in the reference view's clock together with its
uncertainty and source. ADR-0041: arrival time is not exposure time and a
correction is never applied silently — a downstream stage decides whether
the strobe evidence is good enough to use.

The timing report also states the skew each view is expected to accumulate
over the recording from its measured rate deviation, which is what decides
whether one offset per session is sufficient or per-segment alignment is
needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require

TIMING_REPORT_SCHEMA_VERSION = "timing-report/1.0.0"
TIMING_REPORT_FILE = "timing_report.json"
NS_PER_S = 1e9


class ViewTimingReport(BaseModel):
    """One view's alignment evidence over the length of its recording."""

    model_config = ConfigDict(frozen=True)

    view: str
    status: str  # available | unavailable
    offset_ns: int | None = None
    uncertainty_ns: int | None = None
    rate_deviation_ppm: float | None = None
    duration_s: float | None = None
    expected_skew_ms: float | None = None
    reason: str | None = None


class TimingReport(BaseModel):
    """``timing_report.json``."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = TIMING_REPORT_SCHEMA_VERSION
    reference_view: str
    method: str
    clock_domain: str
    status: str
    views: tuple[ViewTimingReport, ...]


def view_timing(timing: Mapping[str, Any], view: str) -> Mapping[str, Any] | None:
    """The ``views`` entry for ``view`` in a manifest ``timing`` block, if any."""
    for entry in timing.get("views", ()):
        if entry.get("view") == view:
            return entry
    return None


def reference_time_fields(
    time_s: float, offset_ns: int, uncertainty_ns: int, source: str
) -> dict[str, float | str]:
    """Fields expressing ``time_s`` in the reference clock, with provenance.

    ``offset_ns`` is this view's arrival-clock offset from the reference view
    (positive = this view stamps later), so the reference-clock time is
    ``time_s - offset``. The original ``time_s`` is left to the caller intact.
    """
    require(uncertainty_ns >= 0, "uncertainty_ns must be non-negative", uncertainty_ns)
    return {
        "time_ref_s": time_s - offset_ns / NS_PER_S,
        "time_ref_uncertainty_s": uncertainty_ns / NS_PER_S,
        "time_ref_source": source,
    }


def annotate_rows(
    rows: list[dict[str, Any]], offset_ns: int, uncertainty_ns: int, source: str
) -> tuple[dict[str, Any], ...]:
    """Return copies of ``rows`` with reference-clock fields added.

    Postcondition: every returned row still carries its original ``time_s``.
    """
    return tuple(
        {
            **row,
            **reference_time_fields(
                float(row["time_s"]), offset_ns, uncertainty_ns, source
            ),
        }
        for row in rows
    )


def _view_report(
    entry: Mapping[str, Any], duration_s: float | None
) -> ViewTimingReport:
    status = str(entry.get("status", "unavailable"))
    deviation = entry.get("rate_deviation_ppm")
    skew_ms = None
    if duration_s is not None and deviation is not None:
        skew_ms = duration_s * float(deviation) / 1e6 * 1e3
    return ViewTimingReport(
        view=str(entry["view"]),
        status=status,
        offset_ns=entry.get("offset_ns"),
        uncertainty_ns=entry.get("uncertainty_ns"),
        rate_deviation_ppm=deviation,
        duration_s=duration_s,
        expected_skew_ms=skew_ms,
        reason=entry.get("reason"),
    )


def build_timing_report(
    timing: Mapping[str, Any], durations: Mapping[str, float]
) -> TimingReport:
    """Fold a manifest ``timing`` block and per-view durations into a report.

    Precondition: ``timing`` has ``reference_view``, ``method``,
    ``clock_domain``, ``status`` and ``views`` (the :class:`sync.TimingRecord`
    shape). Postcondition: one entry per view in ``timing``.
    """
    for key in ("reference_view", "method", "clock_domain", "status", "views"):
        require(key in timing, f"timing block lacks {key}", sorted(timing))
    return TimingReport(
        reference_view=str(timing["reference_view"]),
        method=str(timing["method"]),
        clock_domain=str(timing["clock_domain"]),
        status=str(timing["status"]),
        views=tuple(
            _view_report(entry, durations.get(str(entry.get("view"))))
            for entry in timing["views"]
        ),
    )
