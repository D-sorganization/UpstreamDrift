"""Forward-dynamics marker RMS split at the detected impact (#12042).

The whole-swing FD marker RMS mixes two regimes. Before impact the tracked
reference is stable; after impact the frame-by-frame IK can land on different
near-equal solutions in weakly observed coordinates (toes, shoulder and
forearm axial rotation), and that branch choice moves the whole-swing FD by
several millimetres without any change to the swing itself (see
``docs/research/simscape_matching_reference/simscape_matching_reference.tex``,
"Follow-Through FD Diagnosis"). This module reports the two phases
separately.

It reports only: no gate, threshold or acceptance status reads it.

Impact uses the shared checked rule from GCV-14 (#12004),
:func:`model_appearance.club_face.ball_passage`, applied to the reference's
``Clubhead`` frame. No new impact detector is added.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

#: Name of the impact rule recorded in the receipt.
IMPACT_DETECTOR = "model_appearance.club_face.ball_passage(reference Clubhead)"


def phase_marker_rms(
    times: Sequence[float] | np.ndarray,
    errors: np.ndarray,
    valid: np.ndarray,
    impact_time_s: float,
) -> dict[str, Any]:
    """Valid-marker RMS for address to impact (t <= impact) and after impact.

    Args:
        times: Capture times (frames,), seconds.
        errors: Per-marker FD position error magnitudes (frames, markers), m.
        valid: Marker validity mask (frames, markers).
        impact_time_s: Impact time inside the time span.

    Returns:
        ``fd_rms_address_to_impact_m``, ``fd_rms_after_impact_m`` (``None``
        when a phase has no valid sample) and the frame count of each phase.

    Raises:
        ValueError: on mismatched shapes or an impact outside the time span.
    """
    t = np.asarray(times, dtype=float)
    err = np.asarray(errors, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if err.ndim != 2 or err.shape != mask.shape or err.shape[0] != t.shape[0]:
        raise ValueError("errors, valid and times must share shape (frames, markers)")
    if not t[0] <= impact_time_s <= t[-1]:
        raise ValueError("impact time must lie inside the capture time span")
    pre = t <= impact_time_s

    def rms(rows: np.ndarray) -> float | None:
        sample = err[rows][mask[rows]]
        return float(np.sqrt(np.mean(sample**2))) if sample.size else None

    return {
        "fd_rms_address_to_impact_m": rms(pre),
        "fd_rms_after_impact_m": rms(~pre),
        "address_to_impact_frames": int(pre.sum()),
        "after_impact_frames": int((~pre).sum()),
    }


def fd_phase_report(
    times: Sequence[float] | np.ndarray,
    errors: np.ndarray,
    valid: np.ndarray,
    clubhead: np.ndarray | None,
) -> dict[str, Any]:
    """Receipt block ``dynamics.fd_phase``.

    ``status`` is ``ok`` with the impact time and the two phase RMS values,
    or ``unavailable`` with a reason and ``None`` values; unavailable is
    never reported as zero.
    """
    from src.shared.python.model_appearance.club_face import (  # noqa: PLC0415
        ball_passage,
    )

    base: dict[str, Any] = {
        "impact_detector": IMPACT_DETECTOR,
        "impact_time_s": None,
        "fd_rms_address_to_impact_m": None,
        "fd_rms_after_impact_m": None,
        "address_to_impact_frames": None,
        "after_impact_frames": None,
    }
    if clubhead is None:
        return {**base, "status": "unavailable", "reason": "no reference clubhead"}
    t = np.asarray(times, dtype=float)
    try:
        impact, _, _ = ball_passage(t, np.asarray(clubhead, dtype=float))
    except ValueError as exc:
        return {**base, "status": "unavailable", "reason": str(exc)}
    split = phase_marker_rms(t, errors, valid, impact)
    return {
        **base,
        **split,
        "impact_time_s": float(impact),
        "status": "ok",
        "reason": None,
    }


def reference_clubhead(kin: Any, q_ref: np.ndarray) -> np.ndarray | None:
    """``Clubhead`` frame origin along ``q_ref``, or ``None`` when the
    kinematics cannot place it (for example a lightweight stand-in)."""
    from src.shared.python.motion_matching.pipeline.gaze_residual import (  # noqa: PLC0415
        CLUB_FRAME,
        frame_poses,
    )

    try:
        _, head = frame_poses(kin, q_ref, CLUB_FRAME)
        head = np.asarray(head, dtype=float)
    except (ValueError, TypeError, KeyError):
        return None
    if head.ndim != 2 or head.shape != (len(q_ref), 3):
        return None
    return head
