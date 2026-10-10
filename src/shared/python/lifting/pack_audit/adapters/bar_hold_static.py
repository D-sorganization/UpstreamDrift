"""Engine-free static-equilibrium split of a held bar's weight (LIFT-4, #11744).

Used by the rigid-weld packs (Pinocchio first; OpenSim and Drake follow),
where a rigid two-hand hold of a rigid bar is statically indeterminate and
the engine cannot supply the hand/hand split.  Each engine adapter supplies
the bar's mass and world centre of mass, gravity and the two grip points
from its own model; this module forms the net wrench the hands must exert
and splits it with the shared GCV-7 ``allocate_min_norm``.  No engine
imports, so the contract is tested without any engine installed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    HandWrench,
    allocate_min_norm,
)

__all__ = ["static_hold_split"]

Vec3 = tuple[float, float, float]


def _t3(v: np.ndarray) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def static_hold_split(
    bar_mass_kg: float,
    gravity_mps2: ArrayLike,
    bar_com_m: ArrayLike,
    grip_l_m: ArrayLike,
    grip_r_m: ArrayLike,
) -> dict[str, Any]:
    """Minimum-norm per-hand split of the static-hold wrench on a bar.

    The net wrench the two hands must exert on the bar for static
    equilibrium is ``R = -bar_mass_kg * gravity_mps2`` (the hands carry the
    weight), with the moment of ``R`` about the grip midpoint balancing the
    weight acting at ``bar_com_m`` (gravity has no moment about the bar's
    own centre of mass).  ``allocate_min_norm`` (GCV-7) splits that wrench
    into a minimum-norm pair of hand forces at ``grip_l_m``/``grip_r_m``.

    Postconditions: ``hand_force_n["L"] + hand_force_n["R"] == net_force_n``
    to floating-point precision; a grip symmetric about ``bar_com_m``
    splits the vertical force evenly.

    Raises:
        ValueError: if ``bar_mass_kg`` or the gravity magnitude is not
            positive and finite, any vector is not a finite 3-vector, or the
            grip points coincide.
    """
    if not math.isfinite(bar_mass_kg) or bar_mass_kg <= 0.0:
        raise ValueError(f"bar_mass_kg must be positive and finite, got {bar_mass_kg}")
    vectors = {
        "gravity_mps2": np.asarray(gravity_mps2, dtype=float),
        "bar_com_m": np.asarray(bar_com_m, dtype=float),
        "grip_l_m": np.asarray(grip_l_m, dtype=float),
        "grip_r_m": np.asarray(grip_r_m, dtype=float),
    }
    for name, v in vectors.items():
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must be a finite 3-vector, got {v}")
    gravity = vectors["gravity_mps2"]
    bar_com = vectors["bar_com_m"]
    grip_l = vectors["grip_l_m"]
    grip_r = vectors["grip_r_m"]
    gravity_mag = float(np.linalg.norm(gravity))
    if gravity_mag <= 0.0:
        raise ValueError("gravity magnitude must be positive")
    if float(np.sum((grip_r - grip_l) ** 2)) <= 0.0:
        raise ValueError("left/right grip points coincide; cannot split")

    midpoint = (grip_l + grip_r) / 2.0
    net_force = -bar_mass_kg * gravity
    moment_at_mid = np.cross(bar_com - midpoint, net_force)

    left = HandWrench(side="L", point_m=_t3(grip_l), force_on_club_n=(0.0, 0.0, 0.0))
    right = HandWrench(side="R", point_m=_t3(grip_r), force_on_club_n=(0.0, 0.0, 0.0))
    analysis = GripAnalysis(
        left=left,
        right=right,
        midpoint_m=_t3(midpoint),
        net_force_n=_t3(net_force),
        couple_at_midpoint_nm=_t3(moment_at_mid),
        contact_force_moment_nm=_t3(moment_at_mid),
        applied_free_torque_nm=(0.0, 0.0, 0.0),
        mof_left_nm=None,
        mof_right_nm=None,
        split_method="allocation",
    )
    left_f, right_f = allocate_min_norm(analysis)
    return {
        "midpoint_m": midpoint,
        "net_force_n": net_force,
        "couple_at_midpoint_nm": moment_at_mid,
        "hand_force_n": {
            "L": np.asarray(left_f, dtype=float),
            "R": np.asarray(right_f, dtype=float),
        },
        "gravity_mag_mps2": gravity_mag,
    }
