"""Gear-effect spin of an off-centre strike from rigid two-body contact (#15443).

The increment is the ball spin of the offset strike minus that of the centred
strike, both solved by :func:`physics.rigid_contact.solve_rigid_contact` with
the head's full inertia tensor, so only the strike offset differs. The face is
flat (no bulge or roll) and only the spin is taken: the offset's change to ball
speed and launch direction is not applied. Frame: x down-range, y left, z up;
the offset is (toe, high) from the face centre, toe-positive, as for the legacy
heuristic, which places the heel on the +y side (a right-handed golfer).
"""

from __future__ import annotations

import numpy as np

from src.shared.python.physics.rigid_contact import (
    RigidContactInput,
    solve_rigid_contact,
)

from .types import PreImpactState, RigidGearHead

_UP = np.array([0.0, 0.0, 1.0])
_MIN_HORIZONTAL = 1e-6


def face_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit (normal, heel, face-up) axes; heel = world up x normal.

    Raises:
        ValueError: the normal is not finite or is vertical (no heel axis).
    """
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.all(np.isfinite(n)) or np.linalg.norm(n) == 0.0:
        raise ValueError("face normal must be a finite non-zero 3-vector")
    n = n / np.linalg.norm(n)
    heel = np.cross(_UP, n)
    if np.linalg.norm(heel) < _MIN_HORIZONTAL:
        raise ValueError("face normal is vertical; the heel axis is undefined")
    heel /= np.linalg.norm(heel)
    return n, heel, np.cross(n, heel)


def rigid_gear_spin(
    pre_state: PreImpactState, head: RigidGearHead, cor: float, friction: float
) -> np.ndarray:
    """Ball spin increment [rad/s] (3,) of the strike offset, flat face.

    Preconditions: ``pre_state.impact_offset`` is a finite (toe, high) pair.
    Postcondition: zero for a centred strike.
    """
    if pre_state.impact_offset is None:
        raise ValueError("rigid gear spin needs an impact offset")
    offset = np.asarray(pre_state.impact_offset, dtype=float)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("impact offset must be a finite (toe, high) pair [m]")
    normal, heel, up = face_frame(pre_state.clubhead_orientation)
    frame = np.column_stack((normal, heel, up))
    inertia = frame @ np.diag(head.inertia_diagonal_kg_m2) @ frame.T

    def ball_spin(lever: np.ndarray) -> np.ndarray:
        result = solve_rigid_contact(
            RigidContactInput(
                club_velocity_mps=np.asarray(pre_state.clubhead_velocity, dtype=float),
                club_angular_velocity_rad_s=np.asarray(
                    pre_state.clubhead_angular_velocity, dtype=float
                ),
                ball_velocity_mps=np.asarray(pre_state.ball_velocity, dtype=float),
                ball_angular_velocity_rad_s=np.asarray(
                    pre_state.ball_angular_velocity, dtype=float
                ),
                normal=normal,
                club_lever_m=lever,
                club_inertia_world_kg_m2=inertia,
                club_mass_kg=float(pre_state.clubhead_mass),
                restitution=float(cor),
                friction_coefficient=float(friction),
            )
        )
        return np.asarray(result.ball_angular_velocity_rad_s, dtype=float)

    centre = head.cg_depth_m * normal
    toe, high = offset
    return ball_spin(centre - toe * heel + high * up) - ball_spin(centre)
