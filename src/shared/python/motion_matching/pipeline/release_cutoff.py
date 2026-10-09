"""Release-preserving pre-contact tracking cutoff (GCV-20, #11767).

The tracked reference is low-passed before computed-torque tracking
(DESIGN_DECISIONS section 11) to remove inverse-kinematics noise. A fixed
12 Hz cutoff, however, also removes a fast late release: on the driver capture
the filtered clubhead peaks 13 ms before impact and 10 % slow, while the
unfiltered reference peaks with the capture. The release is signal, not
noise, so the pre-contact cutoff is chosen per reference:

    the lowest candidate cutoff whose pre-contact filtered reference keeps
    the unfiltered reference's clubhead speed-peak time relative to impact
    within one sample and its pre-contact impact speed within
    ``RELEASE_SPEED_TOL``.

Only the reference itself is used (never the capture's speed), so the choice
is a convergence criterion on the filter, not a fit to the acceptance target.
The post-contact samples keep the base cutoff. When no candidate keeps the
release, the widest candidate is used and the result says so.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.shared.python.motion_matching.pipeline.constants import (
    RELEASE_CUTOFF_CANDIDATES_HZ,
    RELEASE_SPEED_TOL,
)
from src.shared.python.motion_matching.pipeline.reference import smooth_reference

ClubheadFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ReleaseCutoff:
    """The selected pre-contact cutoff and the evidence for it."""

    cutoff_hz: float
    base_cutoff_hz: float
    preserved: bool
    unfiltered: dict[str, float]
    timing_tol_s: float
    speed_tol: float
    candidates: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        """JSON-ready receipt block."""
        return {
            "cutoff_hz": self.cutoff_hz,
            "base_cutoff_hz": self.base_cutoff_hz,
            "preserved": self.preserved,
            "unfiltered": dict(self.unfiltered),
            "timing_tol_s": self.timing_tol_s,
            "speed_tol": self.speed_tol,
            "candidates": [dict(row) for row in self.candidates],
        }


def _timing(time: np.ndarray, head: np.ndarray) -> dict[str, float]:
    from src.shared.python.model_appearance.club_face import clubhead_speed_timing

    timing = clubhead_speed_timing(time, head)
    return {
        "peak_minus_impact_s": float(timing.peak_minus_impact_s),
        "impact_speed_mps": float(timing.impact_speed_mps),
    }


def _check_candidates(
    candidates_hz: Sequence[float], base_cutoff_hz: float, nyquist: float
) -> tuple[float, ...]:
    cands = tuple(float(f) for f in candidates_hz)
    if not cands or any(not math.isfinite(f) for f in cands):
        raise ValueError("candidates_hz must be a non-empty sequence of finite cutoffs")
    if any(b <= a for a, b in zip(cands, cands[1:], strict=False)):
        raise ValueError(f"candidates_hz must strictly increase, got {cands}")
    if not (cands[0] > 0.0 and cands[-1] < nyquist):
        raise ValueError(f"candidates_hz must lie in (0, {nyquist}), got {cands}")
    if float(base_cutoff_hz) != cands[0]:
        raise ValueError(
            f"base_cutoff_hz ({base_cutoff_hz}) must be the first candidate {cands[0]}"
        )
    return cands


def release_preserving_cutoff(
    time: Sequence[float] | np.ndarray,
    q: np.ndarray,
    impact_index: int,
    clubhead: ClubheadFn,
    *,
    base_cutoff_hz: float,
    candidates_hz: Sequence[float] = RELEASE_CUTOFF_CANDIDATES_HZ,
    speed_tol: float = RELEASE_SPEED_TOL,
) -> ReleaseCutoff:
    """Lowest candidate pre-contact cutoff that keeps the release.

    Args:
        time: ``(n,)`` strictly increasing, uniformly sampled reference times.
        q: ``(n, nq)`` reference before the tracking low-pass.
        impact_index: Last pre-contact sample (the capture impact split).
        clubhead: Maps a ``(n, nq)`` trajectory to the ``(n, 3)`` world face
            centre (the model's forward kinematics).
        base_cutoff_hz: The post-contact (and default) cutoff; must be the
            first candidate.
        candidates_hz: Increasing cutoffs to try, lowest first.
        speed_tol: Relative impact-speed agreement that keeps a release.

    Returns:
        A :class:`ReleaseCutoff`; ``preserved`` is ``False`` (and the widest
        candidate is returned) when no candidate keeps the release.

    Raises:
        ValueError: bad shapes or time, bad candidates or tolerance, or a
            reference whose clubhead never passes the ball.
    """
    t = np.asarray(time, dtype=float)
    rows = np.asarray(q, dtype=float)
    if rows.ndim != 2 or t.shape != (rows.shape[0],) or t.size < 3:
        raise ValueError("time must be (n,) with one q row per sample")
    step = np.diff(t)
    if np.any(step <= 0.0) or not np.isfinite(t).all():
        raise ValueError("time must be finite and strictly increasing")
    rate_hz = 1.0 / float(np.median(step))
    cands = _check_candidates(candidates_hz, base_cutoff_hz, 0.5 * rate_hz)
    if not (math.isfinite(speed_tol) and speed_tol > 0.0):
        raise ValueError(f"speed_tol must be positive, got {speed_tol}")
    timing_tol_s = float(np.median(step))
    raw = _timing(t, clubhead(rows))
    table: list[dict[str, Any]] = []
    chosen: float | None = None
    for cutoff in cands:
        smooth = smooth_reference(
            rows,
            rate_hz,
            cands[0],
            impact_index=impact_index,
            pre_contact_cutoff_hz=cutoff,
        )
        got = _timing(t, clubhead(smooth))
        keeps = (
            abs(got["peak_minus_impact_s"] - raw["peak_minus_impact_s"])
            <= timing_tol_s * (1.0 + 1e-9)
            and abs(got["impact_speed_mps"] / raw["impact_speed_mps"] - 1.0)
            <= speed_tol
        )
        table.append({"cutoff_hz": cutoff, **got, "preserved": bool(keeps)})
        if keeps and chosen is None:
            chosen = cutoff
    return ReleaseCutoff(
        cutoff_hz=cands[-1] if chosen is None else chosen,
        base_cutoff_hz=cands[0],
        preserved=chosen is not None,
        unfiltered=raw,
        timing_tol_s=timing_tol_s,
        speed_tol=float(speed_tol),
        candidates=tuple(table),
    )
