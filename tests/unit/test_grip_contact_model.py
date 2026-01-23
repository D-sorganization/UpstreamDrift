"""Tests for Contact-Based Grip Model.

Guideline K2 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_contact_model import (
    ContactPoint,
    ContactState,
    GripContactModel,
    GripParameters,
    check_friction_cone,
    classify_contact_state,
    compute_center_of_pressure,
    create_mujoco_grip_contacts,
    decompose_contact_force,
)


class TestFrictionCone:
    """Tests for friction cone calculations."""

    def test_zero_tangent_within_cone(self) -> None:
        """Zero tangential force should be within friction cone."""
        result = check_friction_cone(
            normal_force=100.0,
            tangent_force=np.zeros(3),
            friction_coefficient=0.8,
        )
        assert result

    def test_small_tangent_within_cone(self) -> None:
        """Small tangential force should be within cone."""
        # μ * F_n = 0.8 * 100 = 80 N
        result = check_friction_cone(
            normal_force=100.0,
            tangent_force=np.array([50.0, 0.0, 0.0]),  # 50 N < 80 N
            friction_coefficient=0.8,
        )
        assert result

    def test_large_tangent_outside_cone(self) -> None:
        """Large tangential force should be outside cone (slipping)."""
        # μ * F_n = 0.8 * 100 = 80 N
        result = check_friction_cone(
            normal_force=100.0,
            tangent_force=np.array([100.0, 0.0, 0.0]),  # 100 N > 80 N
            friction_coefficient=0.8,
        )
        assert not result


class TestForceDecomposition:
    """Tests for contact force decomposition."""

    def test_pure_normal_force(self) -> None:
        """Pure normal force should have zero tangent."""
        normal = np.array([0.0, 0.0, 1.0])
        force = np.array([0.0, 0.0, 100.0])

        normal_f, tangent_f = decompose_contact_force(force, normal)

        assert normal_f == pytest.approx(100.0)
        np.testing.assert_allclose(tangent_f, np.zeros(3), atol=1e-10)

    def test_pure_tangent_force(self) -> None:
        """Pure tangential force should have zero normal."""
        normal = np.array([0.0, 0.0, 1.0])
        force = np.array([50.0, 30.0, 0.0])

        normal_f, tangent_f = decompose_contact_force(force, normal)

        assert normal_f == pytest.approx(0.0)
        np.testing.assert_allclose(tangent_f, force, atol=1e-10)

    def test_mixed_force_decomposition(self) -> None:
        """Mixed force should decompose correctly."""
        normal = np.array([0.0, 0.0, 1.0])
        force = np.array([30.0, 40.0, 100.0])

        normal_f, tangent_f = decompose_contact_force(force, normal)

        assert normal_f == pytest.approx(100.0)
        np.testing.assert_allclose(tangent_f, [30.0, 40.0, 0.0], atol=1e-10)


class TestContactClassification:
    """Tests for contact state classification."""

    @pytest.fixture
    def default_params(self) -> GripParameters:
        """Create default grip parameters."""
        return GripParameters()

    def test_no_contact_for_zero_normal(self, default_params: GripParameters) -> None:
        """Zero or negative normal force should be no contact."""
        state = classify_contact_state(
            normal_force=0.0,
            tangent_force=np.zeros(3),
            slip_velocity=np.zeros(3),
            params=default_params,
        )
        assert state == ContactState.NO_CONTACT

    def test_sticking_within_cone(self, default_params: GripParameters) -> None:
        """Contact within friction cone with no slip velocity should stick."""
        state = classify_contact_state(
            normal_force=100.0,
            tangent_force=np.array([10.0, 0.0, 0.0]),  # Well within cone
            slip_velocity=np.zeros(3),
            params=default_params,
        )
        assert state == ContactState.STICKING

    def test_slipping_with_velocity(self, default_params: GripParameters) -> None:
        """Contact with significant slip velocity should be slipping."""
        state = classify_contact_state(
            normal_force=100.0,
            tangent_force=np.array([10.0, 0.0, 0.0]),
            slip_velocity=np.array([0.1, 0.0, 0.0]),  # 0.1 m/s > threshold
            params=default_params,
        )
        assert state == ContactState.SLIPPING


class TestCenterOfPressure:
    """Tests for center of pressure computation."""

    def test_single_contact_cop(self) -> None:
        """COP for single contact should be at contact position."""
        contacts = [
            ContactPoint(
                position=np.array([1.0, 2.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=100.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            )
        ]

        cop = compute_center_of_pressure(contacts)

        np.testing.assert_allclose(cop, [1.0, 2.0, 0.0])

    def test_two_equal_contacts_cop(self) -> None:
        """COP for two equal contacts should be midpoint."""
        contacts = [
            ContactPoint(
                position=np.array([0.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=100.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
            ContactPoint(
                position=np.array([2.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=100.0,
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
        ]

        cop = compute_center_of_pressure(contacts)

        np.testing.assert_allclose(cop, [1.0, 0.0, 0.0])

    def test_weighted_cop(self) -> None:
        """COP should be weighted by normal force."""
        contacts = [
            ContactPoint(
                position=np.array([0.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=100.0,  # Larger force
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
            ContactPoint(
                position=np.array([3.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=50.0,  # Smaller force
                tangent_force=np.zeros(3),
                slip_velocity=np.zeros(3),
                state=ContactState.STICKING,
            ),
        ]

        cop = compute_center_of_pressure(contacts)

        # COP = (100*0 + 50*3) / (100 + 50) = 150/150 = 1.0
        np.testing.assert_allclose(cop, [1.0, 0.0, 0.0])


class TestGripContactModel:
    """Tests for GripContactModel class."""

    def test_update_from_mujoco(self) -> None:
        """Model should update from MuJoCo data."""
        model = GripContactModel()

        # Simulate 2 contact points
        positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        forces = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 50.0]])
        velocities = np.zeros((2, 3))
        body_names = ["left_hand", "left_hand"]

        state = model.update_from_mujoco(
            positions, normals, forces, velocities, body_names, timestamp=0.1
        )

        assert state is not None
        assert len(state.contacts) == 2
        assert state.total_normal_force == pytest.approx(100.0)
        assert state.timestamp == 0.1

    def test_static_equilibrium_check(self) -> None:
        """Model should validate static equilibrium."""
        model = GripContactModel()

        # Create contact that can support 5 N club weight
        positions = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])  # Vertical normal
        forces = np.array([[0.0, 0.0, 5.0]])  # 5 N upward
        velocities = np.zeros((1, 3))
        body_names = ["hand"]

        model.update_from_mujoco(
            positions, normals, forces, velocities, body_names, timestamp=0.0
        )

        result = model.check_static_equilibrium(club_weight=5.0)

        assert result["equilibrium"]
        assert result["support_ratio"] >= 0.99

    def test_slip_margin_calculation(self) -> None:
        """Model should calculate slip margins."""
        model = GripContactModel()

        # Contact well within friction cone
        positions = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        # Pure normal force - zero tangent
        forces = np.array([[0.0, 0.0, 100.0]])
        velocities = np.zeros((1, 3))

        model.update_from_mujoco(
            positions, normals, forces, velocities, ["hand"], timestamp=0.0
        )

        margins = model.check_slip_margin()

        assert margins["min_margin"] == pytest.approx(1.0)  # Full margin available
        assert not margins["any_slipping"]


class TestMuJoCoContactSpec:
    """Tests for MuJoCo contact specification generation."""

    def test_generates_contact_pairs(self) -> None:
        """Should generate contact pairs for each hand."""
        spec = create_mujoco_grip_contacts()

        assert "contact_pairs" in spec
        assert len(spec["contact_pairs"]) == 2  # left and right hand

    def test_custom_body_names(self) -> None:
        """Should use custom body names."""
        spec = create_mujoco_grip_contacts(
            grip_body_name="my_grip",
            hand_body_names=["custom_hand"],
        )

        assert spec["contact_pairs"][0]["body1"] == "custom_hand"
        assert spec["contact_pairs"][0]["body2"] == "my_grip"

    def test_friction_in_spec(self) -> None:
        """Should include friction in contact pairs."""
        spec = create_mujoco_grip_contacts(friction=(0.9, 0.7, 0.002))

        assert spec["contact_pairs"][0]["friction"] == [0.9, 0.7, 0.002]
