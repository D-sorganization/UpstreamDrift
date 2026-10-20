"""Trunk and pelvis series for the HRG-E4 event table (#13175).

Holds the pelvis and thorax world rotations with their provenance, evaluates
the registered lab-frame terms of :mod:`anatomical_angles.trunk_pelvis` (no
term, sign or rotation check is redefined here), and detects the
pelvis-reversal event.

Pelvis reversal is the frame of the maximum of ``pelvis_turn_isb`` (trail
rotation positive) between address and impact: the pelvis stops turning back
and starts turning toward the target. It is the pelvis's own turn reversal,
kept distinct from the frame of maximum X-factor
(``thorax_turn_isb - pelvis_turn_isb``), which usually comes later. A maximum
at either end of the window is no reversal; the event is then reported
undetected with a reason, never placed at a guessed frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.anatomical_angles.trunk_pelvis import (
    PELVIS_INPUT,
    THORAX_INPUT,
    TrunkPelvisTermSpec,
    evaluator,
    term_specs,
)

__all__ = [
    "PELVIS_REVERSAL",
    "PELVIS_REVERSAL_DEFINITION",
    "SINGULAR_REASON",
    "TrunkPelvisSeries",
    "detect_pelvis_reversal",
    "segment_role",
    "trunk_term_degrees",
]

PELVIS_REVERSAL = "pelvis_reversal"
PELVIS_REVERSAL_DEFINITION = (
    "frame of maximum pelvis_turn_isb (trail rotation positive) between address "
    "and impact, inclusive, where the pelvis turn changes from backswing to "
    "downswing; a maximum at a window end is no reversal. Distinct from the "
    "frame of maximum X-factor (thorax_turn_isb - pelvis_turn_isb)"
)
SINGULAR_REASON = (
    "gimbal lock (|lateral bend| = 90 deg): flexion and axial rotation unavailable"
)
_TURN_TERM = "pelvis_turn_isb"


@dataclass(frozen=True)
class TrunkPelvisSeries:
    """Pelvis and thorax ISB world rotations ``(N, 3, 3)`` with provenance.

    ``up`` and ``target`` define the golfer lab frame; ``calibration`` holds
    the pelvis and thorax calibration ids.
    """

    pelvis_rotation_world: object
    thorax_rotation_world: object
    up: object
    target: object
    observability: str
    receipt_path: str
    calibration: tuple[str, str]


def _check_provenance(series: TrunkPelvisSeries) -> None:
    for name in ("observability", "receipt_path"):
        value = getattr(series, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"trunk {name} must be a non-empty string")
    cal = series.calibration
    if len(cal) != 2 or not all(isinstance(c, str) and c for c in cal):
        raise ValueError("trunk calibration must hold the pelvis and thorax ids")


def trunk_term_degrees(
    series: TrunkPelvisSeries, handedness: str, n: int
) -> dict[str, np.ndarray]:
    """Every registered trunk/pelvis term of ``series`` in degrees, by term id.

    Preconditions: provenance strings are non-empty and both rotation series
    have ``n`` samples (each is validated by the registry evaluator). Raises
    ``ValueError`` otherwise. Postcondition: arrays of length ``n``; NaN only
    where the sample is gimbal-locked.
    """
    _check_provenance(series)
    inputs = {
        PELVIS_INPUT: series.pelvis_rotation_world,
        THORAX_INPUT: series.thorax_rotation_world,
        "up": series.up,
        "target": series.target,
    }
    values = {s.term_id: evaluator(s)(inputs, handedness) for s in term_specs()}
    for term_id, arr in values.items():
        if arr.shape != (n,):
            raise ValueError(f"trunk series length for {term_id} differs from time_s")
    return values


def segment_role(spec: TrunkPelvisTermSpec) -> str:
    """``"pelvis"`` or ``"thorax"``: the segment the term describes."""
    return "pelvis" if spec.segment_input == PELVIS_INPUT else "thorax"


def detect_pelvis_reversal(
    pelvis_turn_deg: np.ndarray, start: int, stop: int
) -> tuple[int | None, str]:
    """Pelvis-reversal frame in ``[start, stop]`` and an empty reason.

    Returns ``(None, reason)`` when the window has fewer than three frames, no
    valid sample, or its maximum at an end (NaN samples are ignored).
    """
    if stop - start < 2:
        return None, "address-to-impact window has fewer than three frames"
    window = np.asarray(pelvis_turn_deg, dtype=float)[start : stop + 1]
    if not np.any(np.isfinite(window)):
        return None, f"{_TURN_TERM} has no valid sample between address and impact"
    peak = int(np.nanargmax(window))
    valid = np.flatnonzero(np.isfinite(window))
    if peak in (int(valid[0]), int(valid[-1])):
        return None, (
            f"{_TURN_TERM} peaks at the edge of the address-to-impact window: "
            "no reversal inside it"
        )
    return start + peak, ""
