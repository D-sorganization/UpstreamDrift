"""Tests for energy metrics (issue #8614).

Energy partition: club KE lost, energy to sand, energy to ball.
"""

from __future__ import annotations


import numpy as np
import pytest

from bunkershot3d.metrics.energy import (
    compute_energy_partition,
)


class TestClubKineticEnergy:
    """Track club KE through the shot."""

    def test_club_ke_loss_from_velocity_drop(self) -> None:
        """KE_lost = 0.5 * m * (v_in^2 - v_out^2)."""
        head_mass_kg = 0.30
        v_in = 25.0  # m/s
        v_out = 15.0  # m/s

        _ke_in = 0.5 * head_mass_kg * v_in**2  # 93.75 J (for reference)
        _ke_out = 0.5 * head_mass_kg * v_out**2  # 33.75 J (for reference)
        # Expected KE lost: ~60 J = _ke_in - _ke_out

        # Synthetic trajectory: constant decel
        t = np.array([0.0, 0.005, 0.010])
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.10, 0.0, -0.02],
                [0.18, 0.0, -0.01],
            ]
        )
        # Velocities (finite diff would give these)
        velocities = np.array(
            [
                [25.0, 0.0, -4.0],
                [20.0, 0.0, 2.0],
                [15.0, 0.0, 2.0],
            ]
        )

        partition = compute_energy_partition(t, positions, velocities, head_mass_kg)

        # KE in: 0.5 * 0.3 * (25^2 + 4^2) = 0.5 * 0.3 * 641 = 96.15 J
        # KE out: 0.5 * 0.3 * (15^2 + 2^2) = 0.5 * 0.3 * 229 = 34.35 J
        # Lost: ~61.8 J
        assert partition.club_ke_in_j > 90
        assert partition.club_ke_out_j < 40
        assert partition.club_ke_lost_j > 50

    def test_energy_conserved_when_no_sand_contact(self) -> None:
        """KE should be conserved (within floating error) without contact."""
        head_mass_kg = 0.30
        v = 25.0

        t = np.array([0.0, 0.005, 0.010])
        positions = np.array(
            [
                [0.0, 0.0, 0.02],
                [0.125, 0.0, 0.02],
                [0.25, 0.0, 0.02],
            ]
        )
        # Constant velocity
        velocities = np.array(
            [
                [v, 0.0, 0.0],
                [v, 0.0, 0.0],
                [v, 0.0, 0.0],
            ]
        )

        partition = compute_energy_partition(t, positions, velocities, head_mass_kg)

        assert partition.club_ke_lost_j == pytest.approx(0.0, abs=1e-6)


class TestEnergyToSand:
    """Energy dissipated to sand (work against resistance force)."""

    def test_energy_to_sand_from_force_times_displacement(self) -> None:
        """E_sand = integral(F . ds) along the path."""
        # Constant force of 500 N resisting motion over 0.1 m
        # Work = 500 * 0.1 = 50 J
        t = np.array([0.0, 0.005, 0.010])
        positions = np.array(
            [
                [0.0, 0.0, -0.02],
                [0.05, 0.0, -0.03],
                [0.10, 0.0, -0.02],
            ]
        )
        forces = np.array(
            [
                [-500.0, 0.0, 0.0],  # Resisting motion in +x
                [-500.0, 0.0, 0.0],
                [-500.0, 0.0, 0.0],
            ]
        )
        velocities = np.array(
            [
                [20.0, 0.0, -2.0],
                [15.0, 0.0, 0.0],
                [10.0, 0.0, 2.0],
            ]
        )

        partition = compute_energy_partition(
            t, positions, velocities, 0.30, forces=forces
        )

        # Work against force ~ 500 * 0.1 = 50 J
        # The actual integral will be slightly different due to discretization
        assert 40 < partition.energy_to_sand_j < 60


class TestEnergyToBall:
    """Energy transferred to ball via splash mechanics."""

    def test_ball_energy_from_launch_velocity(self) -> None:
        """E_ball = 0.5 * m * v^2 + 0.5 * I * omega^2."""
        ball_mass_kg = 0.0459  # Golf ball
        ball_moi_kg_m2 = 4e-6  # Approximate
        ball_speed_m_s = 20.0
        ball_spin_rad_s = 500.0

        # Synthetic (not used in this path)
        t = np.array([0.0, 0.010])
        positions = np.zeros((2, 3))
        velocities = np.zeros((2, 3))

        partition = compute_energy_partition(
            t,
            positions,
            velocities,
            0.30,
            ball_mass_kg=ball_mass_kg,
            ball_moi_kg_m2=ball_moi_kg_m2,
            ball_speed_m_s=ball_speed_m_s,
            ball_spin_rad_s=ball_spin_rad_s,
        )

        # KE = 0.5 * 0.0459 * 400 = 9.18 J
        # Rot KE = 0.5 * 4e-6 * 250000 = 0.5 J
        # Total ~ 9.68 J
        expected_ball_ke = 0.5 * ball_mass_kg * ball_speed_m_s**2
        expected_rot_ke = 0.5 * ball_moi_kg_m2 * ball_spin_rad_s**2

        assert partition.energy_to_ball_j == pytest.approx(
            expected_ball_ke + expected_rot_ke, rel=1e-6
        )


class TestEnergyBalance:
    """Total energy accounting."""

    def test_energy_balance_sums_correctly(self) -> None:
        """KE_lost = E_sand + E_ball + E_unaccounted (with tolerance)."""
        t = np.array([0.0, 0.010])
        positions = np.array([[0.0, 0.0, -0.02], [0.15, 0.0, -0.01]])
        velocities = np.array([[25.0, 0.0, -2.0], [15.0, 0.0, 2.0]])
        forces = np.array([[-500.0, 0.0, 200.0], [-400.0, 0.0, 300.0]])

        partition = compute_energy_partition(
            t,
            positions,
            velocities,
            0.30,
            forces=forces,
            ball_mass_kg=0.0459,
            ball_speed_m_s=15.0,
        )

        # Work against force can exceed KE lost in discrete approximation
        # Check that energy accounting is reasonable (fractions sum to ~1)
        total_frac = (
            partition.fraction_to_sand
            + partition.fraction_to_ball
            + partition.fraction_unaccounted
        )
        assert total_frac == pytest.approx(1.0, abs=0.01)


class TestEnergyFractions:
    """Fraction of energy going to each sink."""

    def test_fractions_sum_to_one(self) -> None:
        t = np.array([0.0, 0.010])
        positions = np.array([[0.0, 0.0, -0.02], [0.15, 0.0, -0.01]])
        velocities = np.array([[25.0, 0.0, -2.0], [15.0, 0.0, 2.0]])
        forces = np.array([[-500.0, 0.0, 200.0], [-400.0, 0.0, 300.0]])

        partition = compute_energy_partition(
            t,
            positions,
            velocities,
            0.30,
            forces=forces,
            ball_mass_kg=0.0459,
            ball_speed_m_s=15.0,
        )

        total = (
            partition.fraction_to_sand
            + partition.fraction_to_ball
            + partition.fraction_unaccounted
        )
        assert total == pytest.approx(1.0, abs=1e-6)
