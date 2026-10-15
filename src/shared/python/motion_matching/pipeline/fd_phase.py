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
face centre: the one clubhead measurement point (#12682),
:func:`motion_matching.club_face_target.face_centre_path`. No new impact
detector is added.

The same block carries ``clubhead_speed`` (#12682): the face-centre speed of
the simulated swing, its reference and the capture at their own ball
passages, with the impact-speed error and the peak-timing offset against the
GCV-20 (#11767) tolerances, so every receipt is traceable for speed parity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

#: Name of the impact rule recorded in the receipt.
IMPACT_DETECTOR = "model_appearance.club_face.ball_passage(reference face centre)"
#: Least share of the clubhead's height range above address that must be
#: reached before impact. A real impact follows the backswing; a sample next to
#: address that the detector accepts after a follow-through branch jump does not.
MIN_BACKSWING_RISE_FRACTION = 0.5


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

    ``clubhead`` is the reference face-centre path (:func:`model_face_path`),
    or ``None`` when it cannot be placed. ``status`` is ``ok`` with the impact time and the two phase RMS values,
    or ``unavailable`` with a reason and ``None`` values; unavailable is
    never reported as zero.
    """
    from src.shared.python.model_appearance import club_face  # noqa: PLC0415

    base: dict[str, Any] = {
        "impact_detector": IMPACT_DETECTOR,
        "impact_time_s": None,
        "fd_rms_address_to_impact_m": None,
        "fd_rms_after_impact_m": None,
        "address_to_impact_frames": None,
        "after_impact_frames": None,
    }
    if clubhead is None:
        return {**base, "status": "unavailable", "reason": "no reference face centre"}
    t = np.asarray(times, dtype=float)
    head = np.asarray(clubhead, dtype=float)
    try:
        impact, k, _ = club_face.ball_passage(t, head)
    except ValueError as exc:
        return {**base, "status": "unavailable", "reason": str(exc)}
    rise = head[: k + 1, 2].max() - head[0, 2]
    if rise < MIN_BACKSWING_RISE_FRACTION * (head[:, 2].max() - head[0, 2]):
        return {
            **base,
            "status": "unavailable",
            "reason": (
                f"detected impact at {impact:.3f} s has no backswing before it "
                "(the reference face-centre path is discontinuous)"
            ),
        }
    split = phase_marker_rms(t, errors, valid, impact)
    return {
        **base,
        **split,
        "impact_time_s": float(impact),
        "status": "ok",
        "reason": None,
    }


#: The one clubhead measurement point (#12682): the rendered face centre
#: carried by the ``Clubhead`` frame (``club_face_target.face_centre_path``).
MEASUREMENT_POINT = "face_centre"
#: The one clubhead speed rule (GCV-20, #11767).
SPEED_RULE = "model_appearance.club_face.clubhead_speed_timing"
#: GCV-20 (#11767) parity tolerances: impact speed within 3 % of the
#: capture's, peak within 5 ms of the capture's, each relative to its own
#: impact. Recorded beside the errors; this module does not gate on them.
IMPACT_SPEED_TOL = 0.03
PEAK_TIMING_TOL_S = 0.005
_SPEED_KEYS = (
    "impact_time_s",
    "impact_speed_mps",
    "peak_speed_mps",
    "peak_minus_impact_s",
)


class FacePath(NamedTuple):
    """World face-centre path ``(n, 3)`` at ``times`` ``(n,)``, or ``None``
    centres with the reason it is unavailable."""

    times: np.ndarray | None
    centres: np.ndarray | None
    reason: str | None = None


def model_face_path(
    kin: Any,
    q: np.ndarray,
    times: Sequence[float] | np.ndarray,
    spec: Mapping[str, Any] | None,
) -> FacePath:
    """Face-centre path of ``q`` placed by the IK provider's forward kinematics.

    Unavailable (with the reason) when there is no club specification, the
    kinematics cannot place the ``Clubhead`` frame (a stand-in) or the path
    does not match ``times``. Never falls back to the frame origin.
    """
    from src.shared.python.motion_matching.club_face_target import (  # noqa: PLC0415
        model_face_centres,
    )

    t = np.asarray(times, dtype=float)
    if spec is None:
        return FacePath(t, None, "no club specification")
    try:
        centres = np.asarray(model_face_centres(kin, q, spec), dtype=float)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return FacePath(t, None, f"face centre not placed: {exc}")
    if centres.shape != (len(t), 3):
        return FacePath(t, None, "face centre path does not match the times")
    return FacePath(t, centres)


def capture_face_path(
    times: Sequence[float] | np.ndarray,
    capture: tuple[np.ndarray, np.ndarray, Sequence[str]],
    attachments: Mapping[str, tuple[str, Sequence[float]]] | None,
    spec: Mapping[str, Any] | None,
) -> FacePath:
    """Gap-filled capture face-centre path from the calibrated head triad.

    ``capture`` is the lane's ``(points, valid, labels)`` in the native world.
    Unavailable (with the reason) without the calibrated triad
    ``attachments`` and ``spec``, or when the triad cannot be observed.
    """
    from src.shared.python.motion_matching import club_face_target as cft  # noqa: PLC0415

    t = np.asarray(times, dtype=float)
    if attachments is None or spec is None:
        return FacePath(t, None, "no calibrated head triad")
    points, valid, labels = capture
    try:
        _, centres = cft.observe_capture_face(points, valid, labels, attachments, spec)
        return FacePath(t, cft.fill_unobserved(t, centres))
    except ValueError as exc:
        return FacePath(t, None, f"capture face centre unavailable: {exc}")


def closest_approach_m(
    times: Sequence[float] | np.ndarray, centres: np.ndarray
) -> float:
    """Closest distance (m) of a path to its address point after the top.

    The diagnostic for a face centre that misses the ball; the whole path is
    searched when no backswing top can be located.
    """
    from src.shared.python.model_appearance import club_face  # noqa: PLC0415

    head = np.asarray(centres, dtype=float)
    try:
        top = club_face.top_of_backswing_index(np.asarray(times, dtype=float), head)
    except ValueError:
        top = 0
    return float(np.min(np.linalg.norm(head[top + 1 :] - head[0], axis=1)))


def speed_series(path: FacePath) -> dict[str, Any]:
    """Impact and peak face-centre speed of one path (:data:`SPEED_RULE`).

    ``status`` is ``ok``, or ``unavailable`` with a reason and ``None``
    values (never zero); a path that never passes the ball states its
    closest approach to the ball in the reason.
    """
    from src.shared.python.model_appearance import club_face  # noqa: PLC0415

    base: dict[str, Any] = dict.fromkeys(_SPEED_KEYS)
    if path.centres is None or path.times is None:
        return {**base, "status": "unavailable", "reason": path.reason}
    try:
        timing = club_face.clubhead_speed_timing(path.times, path.centres)
    except ValueError as exc:
        reason = str(exc)
        if "never passes" in reason:
            gap = closest_approach_m(path.times, path.centres)
            reason += f" (closest approach {gap:.3f} m)"
        return {**base, "status": "unavailable", "reason": reason}
    record = timing.to_record()
    return {**{k: record[k] for k in _SPEED_KEYS}, "status": "ok", "reason": None}


def clubhead_speed_report(
    *, simulated: FacePath, reference: FacePath, capture: FacePath
) -> dict[str, Any]:
    """Receipt block ``dynamics.fd_phase.clubhead_speed`` (#12682).

    Face-centre speed of the simulated (forward-dynamics) swing, its tracked
    reference and the capture, each on its own clock relative to its own
    ball passage. ``impact_speed_error_pct`` is the simulated impact speed
    against the capture's, in percent; ``peak_timing_offset_s`` is the
    simulated ``peak_minus_impact_s`` minus the capture's (negative: the
    simulated peak comes earlier). Both are ``None``, with a reason, when
    either side is unavailable. Reporting only: the tolerances are recorded,
    not enforced.
    """
    series = {
        "simulated": speed_series(simulated),
        "reference": speed_series(reference),
        "capture": speed_series(capture),
    }
    report: dict[str, Any] = {
        "measurement_point": MEASUREMENT_POINT,
        "speed_rule": SPEED_RULE,
        "impact_speed_tol_fraction": IMPACT_SPEED_TOL,
        "peak_timing_tol_s": PEAK_TIMING_TOL_S,
        **series,
        "impact_speed_error_pct": None,
        "peak_timing_offset_s": None,
    }
    missing = [k for k in ("simulated", "capture") if series[k]["status"] != "ok"]
    if missing:
        reason = f"no {' or '.join(missing)} face-centre speed"
        return {**report, "status": "unavailable", "reason": reason}
    sim, cap = series["simulated"], series["capture"]
    return {
        **report,
        "impact_speed_error_pct": 100.0
        * (sim["impact_speed_mps"] / cap["impact_speed_mps"] - 1.0),
        "peak_timing_offset_s": sim["peak_minus_impact_s"] - cap["peak_minus_impact_s"],
        "status": "ok",
        "reason": None,
    }
