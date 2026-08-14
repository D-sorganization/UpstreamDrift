"""Energy metrics: KE loss, energy to sand, energy to ball (issue #8614).

Energy partition analysis for understanding where club energy goes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import require

__all__ = [
    "EnergyPartition",
    "compute_energy_partition",
]


@dataclass(frozen=True, slots=True)
class EnergyPartition:
    """Energy accounting for a bunker shot.

    Attributes:
        club_ke_in_j: Club kinetic energy at start [J].
        club_ke_out_j: Club kinetic energy at end [J].
        club_ke_lost_j: Club KE lost during shot [J].
        energy_to_sand_j: Energy dissipated to sand [J].
        energy_to_ball_j: Energy transferred to ball [J].
        energy_unaccounted_j: Unaccounted energy (numerical, sound, etc.) [J].
        fraction_to_sand: Fraction of lost KE going to sand.
        fraction_to_ball: Fraction of lost KE going to ball.
        fraction_unaccounted: Fraction unaccounted for.
    """

    club_ke_in_j: float
    club_ke_out_j: float
    club_ke_lost_j: float
    energy_to_sand_j: float
    energy_to_ball_j: float
    energy_unaccounted_j: float
    fraction_to_sand: float
    fraction_to_ball: float
    fraction_unaccounted: float


def compute_energy_partition(
    t: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    head_mass_kg: float,
    *,
    forces: np.ndarray | None = None,
    ball_mass_kg: float = 0.0,
    ball_moi_kg_m2: float = 0.0,
    ball_speed_m_s: float = 0.0,
    ball_spin_rad_s: float = 0.0,
) -> EnergyPartition:
    """Compute energy partition from trajectory data.

    Args:
        t: Time array (T,) [s].
        positions: Clubhead positions (T, 3) [m].
        velocities: Clubhead velocities (T, 3) [m/s].
        head_mass_kg: Clubhead mass [kg].
        forces: Contact forces (T, 3) [N], optional for work calculation.
        ball_mass_kg: Ball mass [kg].
        ball_moi_kg_m2: Ball moment of inertia [kg.m^2].
        ball_speed_m_s: Ball launch speed [m/s].
        ball_spin_rad_s: Ball spin rate [rad/s].

    Returns:
        EnergyPartition with full energy accounting.
    """
    require(len(t) == len(positions), "t and positions must have same length")
    require(len(t) == len(velocities), "t and velocities must have same length")
    require(head_mass_kg > 0, "head mass must be positive")

    velocities = np.asarray(velocities)
    positions = np.asarray(positions)

    if len(t) < 2:
        return EnergyPartition(
            club_ke_in_j=0.0,
            club_ke_out_j=0.0,
            club_ke_lost_j=0.0,
            energy_to_sand_j=0.0,
            energy_to_ball_j=0.0,
            energy_unaccounted_j=0.0,
            fraction_to_sand=0.0,
            fraction_to_ball=0.0,
            fraction_unaccounted=1.0,
        )

    v_in = velocities[0]
    v_out = velocities[-1]

    ke_in = 0.5 * head_mass_kg * float(np.dot(v_in, v_in))
    ke_out = 0.5 * head_mass_kg * float(np.dot(v_out, v_out))
    ke_lost = max(0.0, ke_in - ke_out)

    e_ball_trans = 0.5 * ball_mass_kg * ball_speed_m_s**2 if ball_mass_kg > 0 else 0.0
    e_ball_rot = (
        0.5 * ball_moi_kg_m2 * ball_spin_rad_s**2 if ball_moi_kg_m2 > 0 else 0.0
    )
    e_ball = e_ball_trans + e_ball_rot

    if forces is not None:
        e_sand = _compute_work_against_force(positions, forces)
    else:
        e_sand = max(0.0, ke_lost - e_ball)

    e_unaccounted = max(0.0, ke_lost - e_sand - e_ball)

    if ke_lost > 0:
        f_sand = e_sand / ke_lost
        f_ball = e_ball / ke_lost
        f_unaccounted = e_unaccounted / ke_lost
    else:
        f_sand = 0.0
        f_ball = 0.0
        f_unaccounted = 1.0 if (e_sand + e_ball + e_unaccounted) == 0 else 0.0

    total = f_sand + f_ball + f_unaccounted
    if total > 0:
        f_sand /= total
        f_ball /= total
        f_unaccounted /= total

    return EnergyPartition(
        club_ke_in_j=ke_in,
        club_ke_out_j=ke_out,
        club_ke_lost_j=ke_lost,
        energy_to_sand_j=e_sand,
        energy_to_ball_j=e_ball,
        energy_unaccounted_j=e_unaccounted,
        fraction_to_sand=f_sand,
        fraction_to_ball=f_ball,
        fraction_unaccounted=f_unaccounted,
    )


def _compute_work_against_force(positions: np.ndarray, forces: np.ndarray) -> float:
    """Compute work done against resistance force: W = -integral(F . ds).

    Negative sign because sand force opposes motion.
    """
    if len(positions) < 2:
        return 0.0

    ds = np.diff(positions, axis=0)
    f_avg = (forces[:-1] + forces[1:]) / 2.0

    work = -np.sum(f_avg * ds)
    return max(0.0, float(work))
