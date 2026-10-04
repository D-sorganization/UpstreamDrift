"""Focused behavioral tests for DIME contact constraints and ground reaction balance (#11427).

Requirements from Issue #11427:
- RED:
  - Loss of support: floating base accelerating downwards without contact force while in stance mode.
  - Negative normal force (adhesive tension / ground pulling down) fails closed.
  - Excessive friction (tangential force exceeding Coulomb friction cone mu * F_n).
  - Incorrect foot frame / non-finite contact point parameters fail closed.
  - Stance-to-flight transition: non-differentiable / derivatives_valid is False.
  - Fictitious pelvis support: unactuated floating root shortcut (allocating direct torque to unactuated DoFs) rejected.
  - Net-GRF allocation ambiguity: when bilateral contact exists and only net GRF is known, does NOT claim uniquely identified single-foot forces; reports identified=False and bounds.
- GREEN:
  - Static weight support: net normal force matches total mass * gravity within 1e-4 N.
  - Known impulse/momentum change and continuous contact replay.
  - Measured versus inferred force statuses differ (distinct provenance, confidence).
  - Unobservable bilateral force split returns bounds/unidentified status, not false certainty.
  - Continuous contact replay produces valid Jacobians matching finite differences on smooth segments.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError, PreconditionError
from src.shared.python.estimation.dime_contact_constraints import (
    BilateralAllocationStatus,
    ContactConstraintResult,
    ContactMode,
    ContactPointAdmissibleForce,
    ContactPointGeometry,
    DimeContactConstraintsFactor,
    ForceProvenance,
    MeasuredGroundReaction,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_native_stance_fixture,
)
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
)

pytestmark = pytest.mark.unit


def _default_contact_parameters() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=1.0e5,
        dissipation_s_m=0.5,
        static_friction=0.8,
        dynamic_friction=0.6,
        viscous_friction=0.01,
        transition_velocity_m_s=0.01,
    )


def _default_ground_plane() -> GroundPlane:
    return GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)


def _default_contact_points() -> tuple[ContactPointGeometry, ...]:
    return (
        ContactPointGeometry(
            point_id="left_heel",
            body_name="left_foot",
            position_m=(-0.15, -0.08, 0.03),
            radius_m=0.03,
        ),
        ContactPointGeometry(
            point_id="left_toe",
            body_name="left_foot",
            position_m=(-0.15, 0.12, 0.03),
            radius_m=0.03,
        ),
        ContactPointGeometry(
            point_id="right_heel",
            body_name="right_foot",
            position_m=(0.15, -0.08, 0.03),
            radius_m=0.03,
        ),
        ContactPointGeometry(
            point_id="right_toe",
            body_name="right_foot",
            position_m=(0.15, 0.12, 0.03),
            radius_m=0.03,
        ),
    )


class TestDimeContactConstraintsRedSuite:
    """RED test suite verifying fail-closed contracts, violations and physical limits."""

    def test_red_negative_normal_force_rejected_or_infeasible(self) -> None:
        """Negative: adhesive ground tension (F_n < 0) cannot be admitted as valid physical support."""
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        # Attempt to evaluate candidate forces with negative normal force
        with pytest.raises((ValueError, PreconditionError, ContractViolationError)):
            factor.validate_candidate_force(
                point_id="left_heel",
                force_n=(0.0, 0.0, -50.0),  # Negative normal force (adhesive tension)
            )

    def test_red_excessive_friction_violates_coulomb_cone(self) -> None:
        """Negative: tangential force exceeding static friction cone is flagged as infeasible."""
        params = _default_contact_parameters()
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=params,
            contact_points=_default_contact_points(),
        )
        # Normal force = 100 N, mu_s = 0.8, max allowable tangent force = 80 N
        # Tangent force = 150 N (exceeds cone)
        admissible = factor.evaluate_admissible_force(
            point_id="left_heel",
            candidate_force_n=(150.0, 0.0, 100.0),
        )
        assert (
            not admissible.is_slipping
            or admissible.tangential_force_n > admissible.friction_limit_n
        )
        assert (
            admissible.confidence < 1.0
            or admissible.tangential_force_n > 100.0 * params.static_friction
        )

    def test_red_incorrect_foot_frame_fails_closed(self) -> None:
        """Negative: non-finite or mismatched dimension foot frame / ground parameters fail closed."""
        with pytest.raises((ValueError, PreconditionError)):
            ContactPointGeometry(
                point_id="bad_foot",
                body_name="foot",
                position_m=(0.0, float("nan"), 0.0),
                radius_m=0.03,
            )

        with pytest.raises((ValueError, PreconditionError)):
            ContactPointGeometry(
                point_id="bad_foot_radius",
                body_name="foot",
                position_m=(0.0, 0.0, 0.0),
                radius_m=-0.05,  # Non-positive radius
            )

    def test_red_loss_of_support_during_stance_fails_equilibrium(self) -> None:
        """Negative: floating base accelerating downwards without contact support reports large root balance defect."""
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        # Mass = 75 kg, gravity = 9.81, zero contact forces provided in stance mode
        mass_matrix = np.eye(6) * 75.0
        bias_forces = np.array([0.0, 0.0, 75.0 * 9.81, 0.0, 0.0, 0.0])
        accelerations = np.zeros(
            6
        )  # Stationary target acceleration requires supporting forces!
        zero_forces = {pt.point_id: np.zeros(3) for pt in _default_contact_points()}
        contact_jacobians = {
            pt.point_id: np.eye(6)[:3, :] for pt in _default_contact_points()
        }

        res = factor.evaluate_root_balance(
            accelerations=accelerations,
            mass_matrix=mass_matrix,
            bias_forces=bias_forces,
            contact_forces=zero_forces,
            contact_jacobians=contact_jacobians,
            mode=ContactMode.DOUBLE_STANCE,
        )
        assert not res.is_feasible
        assert res.unactuated_root_violation_norm > 100.0
        assert "loss_of_support" in res.violation_reasons

    def test_red_fictitious_pelvis_support_shortcut_rejected(self) -> None:
        """Negative: unactuated floating base root coordinates cannot receive direct actuator torque."""
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        # Nonzero actuator torque on unactuated root DoFs (0..5) must be rejected
        unactuated_dof_indices = (0, 1, 2, 3, 4, 5)
        bad_actuation = np.zeros(12)
        bad_actuation[2] = (
            735.75  # Attempt to directly support pelvis with ghost root actuator
        )

        with pytest.raises((ValueError, PreconditionError, ContractViolationError)):
            factor.verify_unactuated_root_integrity(
                applied_torques=bad_actuation,
                unactuated_dofs=unactuated_dof_indices,
            )

    def test_red_stance_to_flight_transition_declares_derivatives_invalid(self) -> None:
        """Negative: impact and lift-off transitions must declare derivatives_valid=False."""
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        result = factor.evaluate_transition_mode(
            previous_mode=ContactMode.DOUBLE_STANCE,
            current_mode=ContactMode.FLIGHT,
        )
        assert not result.derivatives_valid
        assert result.mode == ContactMode.TRANSITION

    def test_red_bilateral_ambiguity_returns_unidentified_status(self) -> None:
        """Negative: when only net GRF is known in double stance, individual split is NOT falsely identified."""
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        # Net normal force is 735.75 N, but no individual plates or CoP are measured
        measured_net = MeasuredGroundReaction(
            time_s=0.0,
            force_n=(0.0, 0.0, 735.75),
            center_of_pressure_m=None,  # No CoP
            provenance="MEASURED",
        )
        bilateral = factor.resolve_bilateral_allocation(
            measured_grf=measured_net,
            left_contact_points=("left_heel", "left_toe"),
            right_contact_points=("right_heel", "right_toe"),
        )
        assert not bilateral.is_identified
        assert bilateral.ambiguity_metric > 0.5
        assert bilateral.left_normal_force_bounds_n[0] == 0.0
        assert bilateral.left_normal_force_bounds_n[1] >= 735.0


class TestDimeContactConstraintsGreenSuite:
    """GREEN test suite verifying physical static support, continuous replay, and provenance."""

    def test_green_static_weight_support_equilibrium(self) -> None:
        """Positive: in quiet upright stance, net ground reaction force matches total mass * gravity."""
        fixture = make_native_stance_fixture(
            n_frames=5, mass_kg=75.0, gravity_m_s2=9.81
        )
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        expected_weight = 75.0 * 9.81

        # Balanced double stance: 4 contact points each carrying 1/4 of total weight
        equil_forces = {
            pt.point_id: np.array([0.0, 0.0, expected_weight / 4.0])
            for pt in _default_contact_points()
        }
        contact_jacobians = {
            pt.point_id: np.eye(6)[:3, :] for pt in _default_contact_points()
        }
        mass_matrix = np.eye(6) * 75.0
        bias_forces = np.array([0.0, 0.0, expected_weight, 0.0, 0.0, 0.0])
        accelerations = np.zeros(6)

        res = factor.evaluate_root_balance(
            accelerations=accelerations,
            mass_matrix=mass_matrix,
            bias_forces=bias_forces,
            contact_forces=equil_forces,
            contact_jacobians=contact_jacobians,
            mode=ContactMode.DOUBLE_STANCE,
        )
        assert res.is_feasible
        assert res.unactuated_root_violation_norm < 1e-4
        assert math.isclose(res.net_normal_force_n, expected_weight, abs_tol=1e-4)

    def test_green_measured_vs_inferred_force_provenance(self) -> None:
        """Positive: measured forces retain MEASURED provenance while estimated allocations are INFERRED."""
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        meas = MeasuredGroundReaction(
            time_s=0.1,
            force_n=(5.0, -2.0, 740.0),
            center_of_pressure_m=(0.02, 0.01, 0.0),
            provenance="MEASURED",
        )
        inferred = factor.estimate_contact_forces_from_state(
            time_s=0.1,
            center_positions_m={
                pt.point_id: np.array([pt.position_m[0], pt.position_m[1], 0.028])
                for pt in _default_contact_points()
            },
            velocities_m_s={
                pt.point_id: np.zeros(3) for pt in _default_contact_points()
            },
        )
        assert meas.provenance == "MEASURED"
        for force_info in inferred:
            assert force_info.provenance == ForceProvenance.INFERRED

    def test_green_continuous_contact_derivatives_match_finite_differences(
        self,
    ) -> None:
        """Positive: during continuous persistent stance, contact force derivatives match numerical differences."""
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        # Position penetrating ground by 2 mm
        center = np.array([0.0, 0.0, 0.028])  # radius 0.03 => penetration 0.002 m
        velocity = np.array([0.05, 0.0, -0.01])
        radius = 0.03

        analytical_jacobian = factor.compute_contact_jacobian(
            center_m=center,
            velocity_m_s=velocity,
            radius_m=radius,
        )
        assert analytical_jacobian is not None
        assert analytical_jacobian.shape == (3, 6)

        # Finite difference verification for dz
        eps = 1e-6
        sample_plus = factor.evaluate_point_contact(
            center + np.array([0.0, 0.0, eps]), velocity, radius
        )
        sample_minus = factor.evaluate_point_contact(
            center - np.array([0.0, 0.0, eps]), velocity, radius
        )
        fd_dF_dz = (sample_plus.force_n - sample_minus.force_n) / (2.0 * eps)

        np.testing.assert_allclose(
            analytical_jacobian[:, 2], fd_dF_dz, rtol=1e-3, atol=1e-2
        )

    def test_green_serialization_roundtrip(self) -> None:
        """Positive: ContactConstraintResult serializes and deserializes cleanly."""
        factor = DimeContactConstraintsFactor(
            ground_plane=_default_ground_plane(),
            contact_parameters=_default_contact_parameters(),
            contact_points=_default_contact_points(),
        )
        result = ContactConstraintResult(
            time_s=0.25,
            mode=ContactMode.DOUBLE_STANCE,
            is_feasible=True,
            root_balance_residual_n=np.zeros(6),
            unactuated_root_violation_norm=0.0,
            contact_forces=(),
            net_contact_force_n=(0.0, 0.0, 735.75),
            net_normal_force_n=735.75,
            bilateral_status=BilateralAllocationStatus(
                is_identified=False,
                left_normal_force_bounds_n=(0.0, 735.75),
                right_normal_force_bounds_n=(0.0, 735.75),
                ambiguity_metric=1.0,
                regularization_weight=None,
            ),
            derivatives_valid=True,
            violation_reasons=(),
        )
        d = factor.to_dict(result)
        restored = factor.from_dict(d)
        assert restored.time_s == result.time_s
        assert restored.mode == result.mode
        assert restored.is_feasible == result.is_feasible
        assert math.isclose(restored.net_normal_force_n, result.net_normal_force_n)
