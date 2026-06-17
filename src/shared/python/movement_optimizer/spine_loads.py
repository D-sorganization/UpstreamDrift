# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Spinal stress estimation at the L5/S1 joint.

Computes compression and anterior-posterior shear forces at the L5/S1
vertebral junction based on the sagittal-plane 3-link model.  The L5/S1
joint is located at the base of the torso segment (hip joint in our model).

Design Principles:
    DBC -- preconditions checked on public functions.
    DRY -- shared mass lookup factored into ``_mass_above_l5``.
    LoD -- callers only use the two public functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import BodyModel

# NIOSH recommended compression limit for occupational lifting (N)
NIOSH_COMPRESSION_LIMIT: float = 3400.0


def _mass_above_l5(body: BodyModel, exercise_type: str) -> float:
    """Return the mass of body segments above L5/S1.

    For squat-type exercises the torso segment already includes arms.
    For deadlift-type exercises the torso segment is trunk+head only.
    """
    if exercise_type in ("squat", "full_squat"):
        return float(body.m_squat[2])
    return float(body.m_deadlift[2])


def _ensure_2d(arr: NDArray) -> NDArray:
    """Ensure the array is 2-D (N, 3).  If 1-D, reshape to (1, 3)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return arr.reshape(1, 3)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (ank, kn, hip), got {arr.shape[1]}")
    return arr


def spinal_compression(
    q: NDArray,
    qd: NDArray,
    qdd: NDArray,
    body: BodyModel,
    bar_mass: float,
    exercise_type: str,
) -> float | NDArray:
    """Compute axial compression force at L5/S1 (N).

    At the L5/S1 joint the compressive force along the spine axis is:

        F_comp = (m_above + bar_mass) * g * cos(torso_angle)
                 + m_torso * qdd_torso * d_com_torso

    Parameters:
        q:  joint angles, shape (3,) or (N, 3)
        qd: joint velocities, shape (3,) or (N, 3)
        qdd: joint accelerations, shape (3,) or (N, 3)
        body: BodyModel instance
        bar_mass: external load in kg
        exercise_type: "squat", "full_squat", "deadlift", etc.

    Returns:
        Compression force in Newtons.  Scalar if input was (3,),
        array of shape (N,) if input was (N, 3).
    """
    scalar_input = np.asarray(q).ndim == 1
    q2d = _ensure_2d(q)
    qdd2d = _ensure_2d(qdd)

    m_above = _mass_above_l5(body, exercise_type)
    torso_angle = q2d[:, 2]

    # Static gravitational component along spine axis
    gravity_comp = (m_above + bar_mass) * body.g * np.cos(torso_angle)

    # Inertial component from torso angular acceleration
    m_torso = (
        float(body.m_squat[2])
        if exercise_type in ("squat", "full_squat")
        else float(body.m_deadlift[2])
    )
    inertial_comp = m_torso * qdd2d[:, 2] * body.d[2]

    result = gravity_comp + inertial_comp

    if scalar_input:
        return float(result[0])
    return result


def spinal_shear(
    q: NDArray,
    qd: NDArray,
    qdd: NDArray,
    body: BodyModel,
    bar_mass: float,
    exercise_type: str,
) -> float | NDArray:
    """Compute anterior-posterior shear force at L5/S1 (N).

    The shear force perpendicular to the spine axis includes both
    gravitational and inertial components:

        F_shear = (m_above + bar_mass) * g * sin(torso_angle)
                  + m_torso * qdd_torso * d_com * cos(torso_angle)
                  + m_torso * qd_torso^2 * d_com * sin(torso_angle)

    The first term is the static gravitational shear.  The second is
    the tangential (angular-acceleration) inertial shear, and the third
    is the centripetal inertial shear.

    Parameters:
        q:  joint angles, shape (3,) or (N, 3)
        qd: joint velocities, shape (3,) or (N, 3)
        qdd: joint accelerations, shape (3,) or (N, 3)
        body: BodyModel instance
        bar_mass: external load in kg
        exercise_type: "squat", "full_squat", "deadlift", etc.

    Returns:
        Shear force in Newtons.  Scalar if input was (3,),
        array of shape (N,) if input was (N, 3).
    """
    scalar_input = np.asarray(q).ndim == 1
    q2d = _ensure_2d(q)
    qd2d = _ensure_2d(qd)
    qdd2d = _ensure_2d(qdd)

    m_above = _mass_above_l5(body, exercise_type)
    torso_angle = q2d[:, 2]

    # Static gravitational shear perpendicular to spine axis
    gravity_shear = (m_above + bar_mass) * body.g * np.sin(torso_angle)

    # Inertial shear from torso angular acceleration and centripetal force
    m_torso = (
        float(body.m_squat[2])
        if exercise_type in ("squat", "full_squat")
        else float(body.m_deadlift[2])
    )
    d_com = body.d[2]

    # Tangential inertial shear: m * qdd * d_com * cos(angle)
    tangential_shear = m_torso * qdd2d[:, 2] * d_com * np.cos(torso_angle)
    # Centripetal inertial shear: m * qd^2 * d_com * sin(angle)
    centripetal_shear = m_torso * qd2d[:, 2] ** 2 * d_com * np.sin(torso_angle)

    result = gravity_shear + tangential_shear + centripetal_shear

    if scalar_input:
        return float(result[0])
    return result
