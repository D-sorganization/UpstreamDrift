"""Wrist and forearm angles of the ``full-body-v1`` model (HRG-E2, #13169).

Engine adapter contract
-----------------------
Any engine supplies the world rotations of three bodies per side (humerus,
forearm, hand), keyed by the short body names of the HRG-00 sign map
(:mod:`motion_matching.coordinate_sign_map`). Body names come from that table,
never from a second list.

Neutral-pose calibration (declared, not measured)
-------------------------------------------------
The specification has no palm, thumb or styloid landmark, so the Wu frames are
calibrated at ``q = 0`` from the HRG-00 convention: the arms are raised forward
with the forearm neutral (thumb toward the elbow-flexion side) and the wrist
neutral. In the model basis (pelvis frame: x forward, y left, z up):

* Wu wrist frame (radius and hand coincide at ``q = 0``): ``Y`` proximal
  ``= -x``; ``X`` volar ``= -y`` on the left (dorsum faces ``+y``) and ``+y``
  on the right (palm faces ``+y``); ``Z = X x Y`` is ``+z`` radial on the right
  and ``-z`` ulnar on the left.
* ISB elbow frame (humerus and forearm coincide at ``q = 0``): ``X`` ``= +z``
  (the side elbow flexion moves the forearm toward), ``Y`` ``= -x`` proximal,
  ``Z = -y`` (right), the same on both sides. Pronation is therefore measured
  from the neutral, thumb-up forearm, not from ISB's supinated anatomical zero.

Each body's constant offset ``C = R_body(q=0)^T W`` absorbs the engine's body
axes and the declared physical grip rotation that sits in the wrist base, so
the hand frame follows the physical hand/club body only. Appearance and visual
grip parameters have no input here. Only the rolled anthropometric wrist family
(``WRIST_ROLL`` applied) carries this convention; see :func:`check_spec_family`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.anatomical_angles.wrist import (
    WristForearm,
    check_side,
    wrist_forearm_angles,
    wu_wrist_frame,
)
from src.shared.python.biomechanics.joint_conventions import AnatomicalFrame
from src.shared.python.motion_matching.coordinate_sign_map import (
    sign_for,
    wrist_roll_applied,
)

#: Prefix of every calibration id produced here.
CALIBRATION_ID = "full-body-v1/hrg00-neutral-q0"
_PROXIMAL = np.array([-1.0, 0.0, 0.0])
_VOLAR = {"L": np.array([0.0, -1.0, 0.0]), "R": np.array([0.0, 1.0, 0.0])}
#: Columns: ISB elbow X, Y, Z in the model basis at ``q = 0`` (both sides).
_ELBOW_MODEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]).T


@dataclass(frozen=True)
class SideChain:
    """Coordinates and short body names of one arm, from the HRG-00 sign map."""

    side: str
    elbow: str
    forearm_axial: str
    wrist_x: str
    wrist_y: str
    humerus: str
    forearm_body: str
    hand: str


def side_chain(side: str) -> SideChain:
    """The arm chain of ``side`` (``"L"``/``"R"``) read from the sign map."""
    check_side(side)
    elbow = sign_for(f"{side}EInput")
    forearm = sign_for(f"{side}FInput")
    wrist_x = sign_for(f"{side}WInputX")
    return SideChain(
        side=side,
        elbow=elbow.coordinate,
        forearm_axial=forearm.coordinate,
        wrist_x=wrist_x.coordinate,
        wrist_y=sign_for(f"{side}WInputY").coordinate,
        humerus=elbow.parent_body,
        forearm_body=wrist_x.parent_body,
        hand=wrist_x.body,
    )


@dataclass(frozen=True)
class NeutralCalibration:
    """Constant body-to-anatomical offsets of one arm (``R_body @ C = frame``)."""

    chain: SideChain
    humerus_elbow: np.ndarray
    forearm_elbow: np.ndarray
    forearm_wrist: np.ndarray
    hand_wrist: np.ndarray

    def frame(self, body_rotation: np.ndarray, offset: np.ndarray, role: str) -> Any:
        """Anatomical frame of a ``(N, 3, 3)`` body series under ``offset``."""
        rot = np.asarray(body_rotation, dtype=float) @ offset
        origin = np.zeros(rot.shape[:-2] + (3,))
        return AnatomicalFrame(
            rot, origin, f"{CALIBRATION_ID}/{self.chain.side}/{role}"
        )


def _body(rotations: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in rotations:
        raise KeyError(f"body rotation for {name!r} is required")
    rot = np.asarray(rotations[name], dtype=float)
    if rot.shape[-2:] != (3, 3) or not np.all(np.isfinite(rot)):
        raise ValueError(f"rotation of {name!r} must be finite with shape (..., 3, 3)")
    return rot


def calibrate_neutral(
    side: str,
    neutral_rotations: Mapping[str, Any],
    *,
    pelvis_rotation: Any | None = None,
) -> NeutralCalibration:
    """Calibrate one arm from the engine's body world rotations at ``q = 0``.

    ``pelvis_rotation`` is the pelvis world rotation at ``q = 0`` (identity if
    omitted); it places the model-basis directions in the engine world.
    Raises ``KeyError`` for a missing body and ``ValueError`` for a malformed
    rotation. Postcondition: at ``q = 0`` every angle is zero.
    """
    chain = side_chain(side)
    pelvis = (
        np.eye(3)
        if pelvis_rotation is None
        else _body({"pelvis": pelvis_rotation}, "pelvis")
    )
    wrist = wu_wrist_frame(
        side, pelvis @ _VOLAR[side], pelvis @ _PROXIMAL, np.zeros(3), CALIBRATION_ID
    ).rotation_world
    elbow = pelvis @ _ELBOW_MODEL
    humerus = _body(neutral_rotations, chain.humerus)
    forearm = _body(neutral_rotations, chain.forearm_body)
    hand = _body(neutral_rotations, chain.hand)
    return NeutralCalibration(
        chain=chain,
        humerus_elbow=humerus.T @ elbow,
        forearm_elbow=forearm.T @ elbow,
        forearm_wrist=forearm.T @ wrist,
        hand_wrist=hand.T @ wrist,
    )


def full_body_v1_wrist_forearm(
    calibration: NeutralCalibration,
    body_rotations: Mapping[str, Any],
    *,
    receipt_path: str,
) -> WristForearm:
    """Wrist (radius to hand) and forearm pronation series for one arm.

    ``body_rotations`` maps the chain's short body names to ``(N, 3, 3)`` world
    rotations of equal length. Every series carries
    ``observability="fitted_grip_rotation"`` and ``receipt_path``.
    """
    chain = calibration.chain
    humerus = _body(body_rotations, chain.humerus)
    forearm = _body(body_rotations, chain.forearm_body)
    hand = _body(body_rotations, chain.hand)
    if not humerus.shape == forearm.shape == hand.shape:
        raise ValueError("humerus, forearm and hand rotation shapes must match")
    return wrist_forearm_angles(
        calibration.frame(humerus, calibration.humerus_elbow, "humerus"),
        calibration.frame(forearm, calibration.forearm_elbow, "forearm"),
        calibration.frame(hand, calibration.hand_wrist, "hand"),
        chain.side,
        receipt_path=receipt_path,
        radius=calibration.frame(forearm, calibration.forearm_wrist, "radius"),
    )


def check_spec_family(spec: Mapping[str, Any], native: Mapping[str, Any]) -> None:
    """``ValueError`` unless ``spec`` is the rolled (``WRIST_ROLL``) family.

    The neutral calibration is written for the anthropometric wrist basis; a
    native spec's second wrist primitive spins about the forearm instead.
    """
    if not wrist_roll_applied(spec, native):
        raise ValueError(
            "the full-body-v1 wrist calibration needs the WRIST_ROLL "
            "(anthropometric) wrist family; this spec keeps the native basis"
        )


__all__ = [
    "CALIBRATION_ID",
    "NeutralCalibration",
    "SideChain",
    "WristForearm",
    "calibrate_neutral",
    "check_spec_family",
    "full_body_v1_wrist_forearm",
    "side_chain",
]
