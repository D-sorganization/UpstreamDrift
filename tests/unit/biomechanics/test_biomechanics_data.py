"""Tests for src.shared.python.biomechanics.biomechanics_data (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.biomechanics.biomechanics_data import BiomechanicalData

# ---------------------------------------------------------------------------
# BiomechanicalData defaults
# ---------------------------------------------------------------------------


class TestBiomechanicalDataDefaults:
    def test_time_default_zero(self) -> None:
        bd = BiomechanicalData()
        assert bd.time == 0.0

    def test_joint_positions_default_empty(self) -> None:
        bd = BiomechanicalData()
        assert len(bd.joint_positions) == 0

    def test_joint_velocities_default_empty(self) -> None:
        bd = BiomechanicalData()
        assert len(bd.joint_velocities) == 0

    def test_joint_torques_default_empty(self) -> None:
        bd = BiomechanicalData()
        assert len(bd.joint_torques) == 0

    def test_club_head_position_default_none(self) -> None:
        bd = BiomechanicalData()
        assert bd.club_head_position is None

    def test_club_head_speed_default_zero(self) -> None:
        bd = BiomechanicalData()
        assert bd.club_head_speed == 0.0

    def test_kinetic_energy_default_zero(self) -> None:
        bd = BiomechanicalData()
        assert bd.kinetic_energy == 0.0

    def test_potential_energy_default_zero(self) -> None:
        bd = BiomechanicalData()
        assert bd.potential_energy == 0.0

    def test_total_energy_default_zero(self) -> None:
        bd = BiomechanicalData()
        assert bd.total_energy == 0.0

    def test_com_position_default_none(self) -> None:
        bd = BiomechanicalData()
        assert bd.com_position is None

    def test_left_foot_force_default_none(self) -> None:
        bd = BiomechanicalData()
        assert bd.left_foot_force is None

    def test_induced_accelerations_default_empty(self) -> None:
        bd = BiomechanicalData()
        assert bd.induced_accelerations == {}

    def test_counterfactuals_default_empty(self) -> None:
        bd = BiomechanicalData()
        assert bd.counterfactuals == {}


# ---------------------------------------------------------------------------
# BiomechanicalData field assignment
# ---------------------------------------------------------------------------


class TestBiomechanicalDataAssignment:
    def test_time_set(self) -> None:
        bd = BiomechanicalData(time=1.5)
        assert bd.time == 1.5

    def test_joint_positions_set(self) -> None:
        q = np.array([1.0, 2.0, 3.0])
        bd = BiomechanicalData(joint_positions=q)
        np.testing.assert_array_equal(bd.joint_positions, q)

    def test_energy_fields(self) -> None:
        bd = BiomechanicalData(
            kinetic_energy=10.0, potential_energy=5.0, total_energy=15.0
        )
        assert bd.kinetic_energy == 10.0
        assert bd.total_energy == 15.0

    def test_club_head_speed_set(self) -> None:
        bd = BiomechanicalData(club_head_speed=50.0)
        assert bd.club_head_speed == 50.0

    def test_com_position_set(self) -> None:
        com = np.array([0.0, 0.0, 1.0])
        bd = BiomechanicalData(com_position=com)
        np.testing.assert_array_equal(bd.com_position, com)

    def test_induced_accelerations_set(self) -> None:
        iaccels = {"gravity": np.array([0.0, 0.0, -9.80665])}
        bd = BiomechanicalData(induced_accelerations=iaccels)
        assert "gravity" in bd.induced_accelerations

    def test_independent_defaults(self) -> None:
        # Two instances should not share mutable defaults
        bd1 = BiomechanicalData()
        bd2 = BiomechanicalData()
        bd1.induced_accelerations["test"] = np.zeros(3)
        assert "test" not in bd2.induced_accelerations
