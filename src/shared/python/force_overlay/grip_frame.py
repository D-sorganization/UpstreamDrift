"""Grip analysis to :class:`ForceTorqueFrame` (GCV-10, #11716, ADR-0052).

One place turns a :class:`~...biomechanics.grip_wrench.GripAnalysis` into the
overlay frame every surface draws (PyQt, web, video).  Glyph labels are
``grip:hand_left``/``grip:hand_right`` (at each grip point), ``grip:net_midpoint``
(force at the midpoint), ``grip:couple_midpoint`` (the equivalent couple) and
``grip:mof_left``/``grip:mof_right``.  A quantity that cannot be computed is not
emitted; its label is listed in ``metadata["grip_unavailable_labels"]`` (never a
zero arrow) together with ``grip_split_method`` so the viewer can name the split.

Kept out of ``force_overlay/__init__`` because the biomechanics core imports
this package.
"""

from __future__ import annotations

import math
from typing import Any

from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    to_overlay_wrenches,
)
from src.shared.python.force_overlay.contracts import ForceTorqueFrame, OverlayWrench

__all__ = ["GRIP_LABELS", "grip_frame", "grip_wrenches_and_metadata"]

#: Every label a complete grip analysis can emit.
GRIP_LABELS: tuple[str, ...] = (
    "grip:hand_left",
    "grip:hand_right",
    "grip:net_midpoint",
    "grip:couple_midpoint",
    "grip:mof_left",
    "grip:mof_right",
)


def grip_wrenches_and_metadata(
    analysis: GripAnalysis, *, source: str
) -> tuple[tuple[OverlayWrench, ...], dict[str, Any]]:
    """Overlay wrenches and frame metadata of one grip analysis.

    Metadata: ``grip_split_method``, ``grip_unavailable_labels``, optionally
    ``grip_unavailable_reason`` and ``grip_midpoint_m`` (the camera focus of the
    hands close-up view).

    Raises:
        TypeError: if ``analysis`` is not a ``GripAnalysis``.
    """
    if not isinstance(analysis, GripAnalysis):
        raise TypeError(f"analysis must be a GripAnalysis, got {type(analysis)}")
    wrenches = tuple(to_overlay_wrenches(analysis, source=source))
    present = {w.label for w in wrenches}
    metadata: dict[str, Any] = {
        "grip_split_method": analysis.split_method,
        "grip_unavailable_labels": [x for x in GRIP_LABELS if x not in present],
    }
    if analysis.unavailable_reason:
        metadata["grip_unavailable_reason"] = analysis.unavailable_reason
    if analysis.midpoint_m is not None:
        metadata["grip_midpoint_m"] = list(analysis.midpoint_m)
    return wrenches, metadata


def grip_frame(
    time_s: float,
    analysis: GripAnalysis,
    *,
    engine: str,
    source: str,
    extra_metadata: dict[str, Any] | None = None,
) -> ForceTorqueFrame:
    """Overlay frame of one grip analysis.

    Raises:
        TypeError: if ``analysis`` is not a ``GripAnalysis``.
        ValueError: for a non-finite ``time_s`` or an empty ``engine``/``source``.
    """
    if not math.isfinite(time_s):
        raise ValueError("time_s must be finite")
    if not engine or not source:
        raise ValueError("engine and source must be non-empty")
    wrenches, metadata = grip_wrenches_and_metadata(analysis, source=source)
    metadata = {**(extra_metadata or {}), **metadata}
    return ForceTorqueFrame(
        time_s=float(time_s), engine=engine, wrenches=wrenches, metadata=metadata
    )
