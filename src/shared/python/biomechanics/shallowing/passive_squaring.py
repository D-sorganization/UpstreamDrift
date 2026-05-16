"""Passive squaring torque and swing-plane classification (Phase 2, epic #5422).

Implements the centrifugal-torque model from MacKenzie (2012): a club centre
of mass that lies *off* the hand-path plane experiences a centripetal
acceleration whose moment about the wrist behaves as an external torque.
This torque passively rotates the club face toward (or away from) square,
depending on the sign of the perpendicular offset.

Sign conventions
----------------
The hand-path plane normal is oriented upward (``normal[2] >= 0`` per
:mod:`hand_path_plane`).  A *positive* offset means the club CoM is on the
upper (steepening) side of the plane.  A *negative* offset means it is on
the lower (shallowing) side.  The passive torque carries the same sign as
the offset so that downstream code can simply add it to the active wrist
torque.

Design by Contract
------------------
- :func:`compute_club_com_offset`: ``club_com_3d`` must be shape ``(3,)``;
  ``hand_path_plane.normal`` should be unit length (we tolerate non-unit
  input by normalising internally rather than raising).
- :func:`compute_passive_squaring_torque`: ``club_mass`` must be strictly
  positive; raises :class:`ValueError` otherwise.
- :func:`classify_swing_plane`: threshold of 0.02 m on
  ``|steepness_index - shallowing_index|`` separates ``on_plane`` from the
  steep/shallow labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .hand_path_plane import Plane3D

_LOG = logging.getLogger(__name__)

# Threshold (metres) below which the swing is considered to be on-plane.
_ON_PLANE_THRESHOLD_M: float = 0.02


def compute_club_com_offset(club_com_3d: np.ndarray, hand_path_plane: Plane3D) -> float:
    """Signed perpendicular distance from the club CoM to the hand-path plane.

    Positive values indicate the CoM lies above the plane (steepening
    contribution); negative values indicate it lies below (shallowing
    contribution).  The sign convention follows the upward-hemisphere
    orientation enforced by :func:`compute_hand_path_plane`.

    Design by Contract:
        Preconditions:
            - ``club_com_3d`` must be a NumPy array of shape ``(3,)``.
            - ``hand_path_plane.normal`` should be unit length; non-unit
              vectors are normalised internally (with a warning).
        Postconditions:
            - Returns a finite ``float``.

    Args:
        club_com_3d: 3-D position of the club centre of mass (metres).
        hand_path_plane: Best-fit plane through the lead-hand trajectory.

    Returns:
        Signed perpendicular distance in metres.

    Raises:
        ValueError: If ``club_com_3d`` is not shape ``(3,)``.
        ValueError: If ``hand_path_plane.normal`` has zero length.
    """
    com = np.asarray(club_com_3d, dtype=float)
    if com.shape != (3,):
        raise ValueError(f"club_com_3d must have shape (3,), got shape {com.shape}")

    normal = np.asarray(hand_path_plane.normal, dtype=float)
    norm_len = float(np.linalg.norm(normal))
    if norm_len == 0.0:
        raise ValueError("hand_path_plane.normal must be non-zero")
    if not np.isclose(norm_len, 1.0):
        _LOG.debug(
            "hand_path_plane.normal not unit length (%.6f); normalising", norm_len
        )
        normal = normal / norm_len

    offset = float(np.dot(com - hand_path_plane.point_on_plane, normal))
    return offset


def compute_passive_squaring_torque(
    com_offset: float, angular_velocity: float, club_mass: float
) -> float:
    """External torque on the wrist from centrifugal force on out-of-plane CoM.

    Model:
        ``tau = club_mass * angular_velocity**2 * com_offset``

    The torque magnitude scales with the square of the angular velocity
    (centripetal acceleration) and the perpendicular offset of the CoM.  The
    sign tracks ``com_offset`` so that a positive offset (CoM above the
    plane) yields a positive restoring torque toward the plane.

    Design by Contract:
        Preconditions:
            - ``club_mass`` must be strictly positive.
            - ``angular_velocity`` may be any real number (signed angular
              speed).
            - ``com_offset`` may be any real number.
        Postconditions:
            - Returns a finite ``float`` in N*m.
            - ``sign(tau) == sign(com_offset)`` when ``com_offset != 0``.

    Args:
        com_offset: Signed perpendicular distance of CoM from plane (m).
        angular_velocity: Angular speed of the club about the wrist (rad/s).
        club_mass: Mass of the club (kg).

    Returns:
        Passive squaring torque in newton-metres.

    Raises:
        ValueError: If ``club_mass <= 0``.
    """
    if club_mass <= 0.0:
        raise ValueError(f"club_mass must be strictly positive, got {club_mass!r}")
    return float(club_mass * (angular_velocity**2) * com_offset)


@dataclass(frozen=True)
class ShallowingMetrics:
    """Summary metrics describing shallowing behaviour across a full swing.

    Attributes:
        com_offset: Instantaneous signed offset at a representative frame
            (e.g. mid-downswing), metres.
        passive_torque: Instantaneous passive squaring torque at the same
            frame, newton-metres.
        steepness_index: Maximum *positive* offset observed over the swing,
            metres.  Captures the strongest steepening excursion.
        shallowing_index: Maximum *magnitude* of the negative offset observed
            over the swing, metres.  Captures the strongest shallowing
            excursion (always non-negative).
    """

    com_offset: float
    passive_torque: float
    steepness_index: float
    shallowing_index: float


def classify_swing_plane(
    metrics: ShallowingMetrics,
) -> Literal["steep", "on_plane", "shallow"]:
    """Classify a swing as steep, on-plane, or shallow.

    The two indices are compared:

    - If ``|steepness_index - shallowing_index| < 0.02 m`` the swing is
      ``on_plane``.
    - Otherwise the larger index wins: ``steep`` if
      ``steepness_index > shallowing_index``, else ``shallow``.

    Design by Contract:
        Postconditions:
            - Returns one of the literal strings ``"steep"``, ``"on_plane"``,
              ``"shallow"``.

    Args:
        metrics: Aggregated shallowing metrics over the swing.

    Returns:
        Classification label as a string literal.
    """
    delta = abs(metrics.steepness_index - metrics.shallowing_index)
    if delta < _ON_PLANE_THRESHOLD_M:
        return "on_plane"
    if metrics.steepness_index > metrics.shallowing_index:
        return "steep"
    return "shallow"
