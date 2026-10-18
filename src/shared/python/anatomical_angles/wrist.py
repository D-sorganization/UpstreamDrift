"""Wrist and forearm anatomical angles from calibrated segment frames.

Conventions (Wu et al. 2005, ISB part II, section 4.3 and 3.4.1)
----------------------------------------------------------------
* Wrist: radius frame to third-metacarpal (hand) frame, Cardan ``ZXY``
  (profile ``isb_wrist``): flexion about the radius Z, ulnar deviation about
  the floating axis, and a *third* angle about the hand long axis. That third
  angle is a wrist axial rotation; it is NOT forearm pronation.
* Forearm pronation/supination: humerus to forearm, ISB elbow ``ZXY``
  (profile ``isb_elbow``), third angle. Elbow frames differ from the wrist
  frames, so pronation is reported as its own series.
* Per-side frames. Both sides use the proper frame ``X`` volar, ``Y`` proximal,
  ``Z = X x Y`` (radial on the right, ulnar on the left). Raw Cardan angles of
  the left side flip sign for the second and third components; the
  :data:`SIDE_SIGN` table maps them back to anatomical senses so a mirrored
  motion gives identical values: flexion, ulnar deviation and pronation are
  positive on both sides.
* Singularities. At ``|second angle| = 90 deg`` the first and third angles are
  indeterminate: those components are NaN with a reason; the second remains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.biomechanics.joint_conventions import (
    AnatomicalFrame,
    anatomical_joint_angles,
)

Side = Literal["L", "R"]
#: Wrist angles come from a fitted grip rotation, not measured hand anatomy.
OBSERVABILITY = "fitted_grip_rotation"
#: Raw ZXY components -> anatomical-sign components, per side (see module doc).
SIDE_SIGN: dict[str, np.ndarray] = {
    "R": np.array([1.0, 1.0, 1.0]),
    "L": np.array([1.0, -1.0, -1.0]),
}
_GIMBAL = "gimbal lock: {} indeterminate when |{}| = 90 deg"


def check_side(side: object) -> str:
    """``side`` if ``"L"`` or ``"R"``; ``ValueError`` otherwise."""
    if side not in SIDE_SIGN:
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    return str(side)


def check_receipt(receipt_path: object) -> str:
    """Non-empty receipt path; ``ValueError`` otherwise."""
    if not isinstance(receipt_path, str) or not receipt_path.strip():
        raise ValueError("a receipt_path is required for model-estimated angles")
    return receipt_path


@dataclass(frozen=True)
class _Decomposed:
    angles: np.ndarray  # (N, 3) rad, anatomical sign, NaN where indeterminate
    reasons: tuple[np.ndarray, np.ndarray, np.ndarray]
    relative: np.ndarray  # (N, 3, 3)


def _decompose(
    proximal: AnatomicalFrame,
    distal: AnatomicalFrame,
    profile: str,
    side: str,
    names: tuple[str, str, str],
) -> _Decomposed:
    """Relative rotation -> ISB profile angles, side-signed, singular -> NaN."""
    try:  # a single fixed frame broadcasts against a series
        rel = np.swapaxes(proximal.rotation_world, -1, -2) @ distal.rotation_world
    except ValueError as exc:
        raise ValueError("proximal and distal frame shapes must match") from exc
    rel = rel.reshape(-1, 3, 3)
    result = anatomical_joint_angles(rel, profile)
    angles = np.array(result.values, dtype=float).reshape(-1, 3) * SIDE_SIGN[side]
    singular = np.asarray(result.singular, dtype=bool).reshape(-1)
    reasons = tuple(np.full(len(angles), "", dtype=object) for _ in range(3))
    for k in (0, 2):
        angles[singular, k] = np.nan
        reasons[k][singular] = _GIMBAL.format(names[k], names[1])
    return _Decomposed(angles, reasons, rel)  # type: ignore[arg-type]


def decompose_relative(
    relative_rotation: object, profile: str, side: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(angles_rad, singular)`` of ``(N, 3, 3)`` relative rotations.

    Used by the golf-term registry, which receives the relative rotation
    directly. Indeterminate components are NaN, never zero.
    """
    check_side(side)
    rel = np.asarray(relative_rotation, dtype=float)
    origin = np.zeros(rel.shape[:-2] + (3,))
    ident = AnatomicalFrame(np.broadcast_to(np.eye(3), rel.shape), origin, "rel")
    frame = AnatomicalFrame(rel, origin, "rel")
    d = _decompose(ident, frame, profile, side, ("a", "b", "c"))
    singular = np.array([bool(r) for r in d.reasons[0]])
    return d.angles, singular


@dataclass(frozen=True)
class WristAngles:
    """Wrist series (radians; ``*_deg`` properties give degrees).

    ``flexion`` positive = flexion; ``ulnar_deviation`` positive = ulnar;
    ``axial_rotation`` is the third wrist angle about the hand long axis,
    positive in the pronation sense. ``reasons[name][i]`` is empty for a valid
    sample, else why it is NaN. ``observability`` and ``receipt_path`` travel
    with the series.
    """

    flexion_rad: np.ndarray
    ulnar_deviation_rad: np.ndarray
    axial_rotation_rad: np.ndarray
    reasons: dict[str, np.ndarray]
    side: str
    receipt_path: str
    calibration: tuple[str, str]
    observability: str = field(default=OBSERVABILITY)

    @property
    def flexion_deg(self) -> np.ndarray:
        return np.rad2deg(self.flexion_rad)

    @property
    def ulnar_deviation_deg(self) -> np.ndarray:
        return np.rad2deg(self.ulnar_deviation_rad)

    @property
    def axial_rotation_deg(self) -> np.ndarray:
        return np.rad2deg(self.axial_rotation_rad)


@dataclass(frozen=True)
class ForearmAngles:
    """Elbow-profile series: flexion, carrying angle and forearm pronation."""

    flexion_rad: np.ndarray
    carrying_angle_rad: np.ndarray
    pronation_rad: np.ndarray
    reasons: dict[str, np.ndarray]
    side: str
    receipt_path: str
    calibration: tuple[str, str]
    observability: str = field(default=OBSERVABILITY)

    @property
    def pronation_deg(self) -> np.ndarray:
        return np.rad2deg(self.pronation_rad)


@dataclass(frozen=True)
class WristForearm:
    """Wrist and forearm series reported side by side, never merged."""

    wrist: WristAngles
    forearm: ForearmAngles


def wrist_angles(
    radius: AnatomicalFrame,
    hand: AnatomicalFrame,
    side: str,
    *,
    receipt_path: str,
) -> WristAngles:
    """Wrist angles of ``hand`` in ``radius`` for ``side``.

    Preconditions: both frames are Wu wrist frames (see :func:`wu_wrist_frame`)
    of equal batch shape; ``side`` is ``"L"``/``"R"``; ``receipt_path`` names
    the fit receipt. Postcondition: three arrays of equal length with NaN (and
    a reason) only for the indeterminate components of a singular sample.
    """
    check_side(side)
    check_receipt(receipt_path)
    d = _decompose(
        radius, hand, "isb_wrist", side, ("flexion", "ulnar_deviation", "axial")
    )
    names = ("flexion", "ulnar_deviation", "axial_rotation")
    return WristAngles(
        d.angles[:, 0].copy(),
        d.angles[:, 1].copy(),
        d.angles[:, 2].copy(),
        dict(zip(names, d.reasons, strict=True)),
        side,
        receipt_path,
        (radius.calibration_id, hand.calibration_id),
    )


def forearm_pronation(
    humerus: AnatomicalFrame,
    forearm: AnatomicalFrame,
    side: str,
    *,
    receipt_path: str,
) -> ForearmAngles:
    """Forearm pronation (ISB elbow third angle) of ``forearm`` in ``humerus``.

    Elbow frames are the ISB elbow frames (X anterior, Y proximal, Z right at
    the neutral pose, same axes both sides); left-side signs follow
    :data:`SIDE_SIGN`. Reported separately from the wrist's third angle.
    """
    check_side(side)
    check_receipt(receipt_path)
    d = _decompose(humerus, forearm, "isb_elbow", side, ("flexion", "carrying", "pro"))
    names = ("flexion", "carrying_angle", "pronation")
    return ForearmAngles(
        d.angles[:, 0].copy(),
        d.angles[:, 1].copy(),
        d.angles[:, 2].copy(),
        dict(zip(names, d.reasons, strict=True)),
        side,
        receipt_path,
        (humerus.calibration_id, forearm.calibration_id),
    )


def wrist_forearm_angles(
    humerus: AnatomicalFrame,
    forearm: AnatomicalFrame,
    hand: AnatomicalFrame,
    side: str,
    *,
    receipt_path: str,
    radius: AnatomicalFrame | None = None,
) -> WristForearm:
    """Wrist (radius -> hand) and forearm pronation (humerus -> forearm).

    ``radius`` defaults to ``forearm`` when the forearm and radius frames are
    the same calibrated segment.
    """
    return WristForearm(
        wrist_angles(
            forearm if radius is None else radius,
            hand,
            side,
            receipt_path=receipt_path,
        ),
        forearm_pronation(humerus, forearm, side, receipt_path=receipt_path),
    )


def reconstruct_wrist_rotation(angles: WristAngles) -> np.ndarray:
    """Relative rotations ``(N, 3, 3)`` rebuilt from a finite wrist series.

    ``ValueError`` when any component is NaN: an indeterminate sample cannot be
    reconstructed from its angles.
    """
    a = np.column_stack(
        [angles.flexion_rad, angles.ulnar_deviation_rad, angles.axial_rotation_rad]
    )
    if np.isnan(a).any():
        raise ValueError("cannot reconstruct a rotation from NaN components")
    raw = a * SIDE_SIGN[angles.side]
    return np.asarray(Rotation.from_euler("ZXY", raw).as_matrix())


def wu_wrist_frame(
    side: str,
    volar_world: object,
    proximal_world: object,
    origin_world: object,
    calibration_id: str,
) -> AnatomicalFrame:
    """Proper Wu wrist frame from the segment's volar and proximal directions.

    ``X`` = volar, ``Y`` = proximal (orthogonalised against ``X``),
    ``Z = X x Y``: radial on the right, ulnar on the left. Calibration
    directions come from the landmark/neutral-pose calibration, not from engine
    body axes. ``ValueError`` for zero or parallel directions.
    """
    check_side(side)
    x = np.asarray(volar_world, dtype=float)
    y = np.asarray(proximal_world, dtype=float)
    if np.linalg.norm(x) < 1e-9 or np.linalg.norm(y) < 1e-9:
        raise ValueError("volar and proximal directions must be nonzero")
    x = x / np.linalg.norm(x)
    y = y - float(x @ y) * x
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("volar and proximal directions must not be parallel")
    y = y / np.linalg.norm(y)
    rot = np.column_stack([x, y, np.cross(x, y)])
    return AnatomicalFrame(rot, np.asarray(origin_world, float), calibration_id)
