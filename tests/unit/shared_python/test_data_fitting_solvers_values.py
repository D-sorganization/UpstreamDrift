"""Value-asserting tests for data-fitting solvers (#6999).

Replaces smoke-only coverage of IK + parameter estimation in
``validation_pkg._data_fitting_solvers`` with assertions that recovered
parameters / poses match known synthetic ground truth within tolerance,
plus convergence and bad-input (DbC) contracts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.validation_pkg.data_fitting import (
    InverseKinematicsSolver,
    ParameterEstimator,
)
from src.shared.python.validation_pkg.data_fitting import (
    KinematicState,
)


@pytest.fixture
def two_link_solver() -> InverseKinematicsSolver:
    """A 2-joint planar chain with unit segment lengths."""
    return InverseKinematicsSolver(
        segment_lengths={"a": 1.0, "b": 1.0},
        joint_names=["a_joint", "b_joint"],
        tolerance=1e-10,
        max_iterations=200,
    )


def _two_link_endpoint(
    t1: float, t2: float, l1: float, l2: float
) -> tuple[float, float]:
    """Ground-truth forward map of the standard 2-link planar arm."""
    x = l1 * math.cos(t1) + l2 * math.cos(t1 + t2)
    y = l1 * math.sin(t1) + l2 * math.sin(t1 + t2)
    return x, y


# ---------------------------------------------------------------------------
# solve_analytical_2d: reachable -> FK round-trip; unreachable -> ValueError
# ---------------------------------------------------------------------------
class TestAnalytical2D:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("t1", "t2"),
        [(0.5, 0.7), (0.1, 1.2), (-0.4, 0.9), (0.8, 0.3)],
    )
    def test_round_trip_recovers_pose(
        self, two_link_solver: InverseKinematicsSolver, t1: float, t2: float
    ) -> None:
        l1 = l2 = 1.0
        x, y = _two_link_endpoint(t1, t2, l1, l2)
        s1, s2 = two_link_solver.solve_analytical_2d(np.array([x, y]), l1, l2)
        # Forward-map the recovered angles and confirm they hit the target.
        rx, ry = _two_link_endpoint(s1, s2, l1, l2)
        assert rx == pytest.approx(x, abs=1e-9)
        assert ry == pytest.approx(y, abs=1e-9)

    @pytest.mark.unit
    def test_unreachable_far_raises(
        self, two_link_solver: InverseKinematicsSolver
    ) -> None:
        with pytest.raises(ValueError, match="unreachable"):
            two_link_solver.solve_analytical_2d(np.array([5.0, 0.0]), 1.0, 1.0)

    @pytest.mark.unit
    def test_unreachable_too_close_raises(
        self, two_link_solver: InverseKinematicsSolver
    ) -> None:
        # |L1 - L2| = 1.0 with very unequal links; target inside inner radius.
        with pytest.raises(ValueError, match="too close"):
            two_link_solver.solve_analytical_2d(np.array([0.1, 0.0]), 2.0, 1.0)

    @pytest.mark.unit
    def test_fully_extended_reaches_sum(
        self, two_link_solver: InverseKinematicsSolver
    ) -> None:
        # Target exactly at max reach -> elbow straight (theta2 ~ 0).
        _, theta2 = two_link_solver.solve_analytical_2d(np.array([2.0, 0.0]), 1.0, 1.0)
        assert theta2 == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _forward_kinematics: known geometry
# ---------------------------------------------------------------------------
class TestForwardKinematics:
    @pytest.mark.unit
    def test_zero_angles_straight_line(
        self, two_link_solver: InverseKinematicsSolver
    ) -> None:
        positions = two_link_solver._forward_kinematics(np.array([0.0, 0.0]))
        # Two unit segments along +x: endpoints at (1,0,0) and (2,0,0).
        assert positions[0] == pytest.approx([1.0, 0.0, 0.0])
        assert positions[-1] == pytest.approx([2.0, 0.0, 0.0])

    @pytest.mark.unit
    def test_right_angle_first_joint(
        self, two_link_solver: InverseKinematicsSolver
    ) -> None:
        positions = two_link_solver._forward_kinematics(np.array([math.pi / 2, 0.0]))
        # First segment points +y -> (0,1); second continues +y -> (0,2).
        assert positions[0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
        assert positions[-1] == pytest.approx([0.0, 2.0, 0.0], abs=1e-12)


# ---------------------------------------------------------------------------
# solve_numerical: converges to synthetic ground truth within tolerance
# ---------------------------------------------------------------------------
class TestSolveNumerical:
    @pytest.mark.unit
    def test_converges_to_known_angles(
        self, two_link_solver: InverseKinematicsSolver
    ) -> None:
        true_angles = np.array([0.3, 0.4])
        targets = two_link_solver._forward_kinematics(true_angles)
        result = two_link_solver.solve_numerical(
            targets, initial_angles=np.array([0.1, 0.1])
        )
        assert result.success
        # Cumulative-angle FK is invariant to a shared offset only in pure
        # rotation; the residual (and thus reproduced positions) must vanish.
        assert result.rms_error == pytest.approx(0.0, abs=1e-6)
        reproduced = two_link_solver._forward_kinematics(
            np.array([result.parameters["a_joint"], result.parameters["b_joint"]])
        )
        np.testing.assert_allclose(reproduced, targets, atol=1e-6)

    @pytest.mark.unit
    def test_bad_input_raises(self, two_link_solver: InverseKinematicsSolver) -> None:
        with pytest.raises(ValueError):
            two_link_solver.solve_numerical(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ParameterEstimator: lengths + anthropometric params vs known fixtures
# ---------------------------------------------------------------------------
class TestParameterEstimator:
    @pytest.mark.unit
    def test_estimate_segment_length_constant(self) -> None:
        est = ParameterEstimator("dempster")
        prox = np.zeros((5, 3))
        dist = np.tile(np.array([0.3, 0.0, 0.0]), (5, 1))
        mean_len, std_len = est.estimate_segment_length(prox, dist)
        assert mean_len == pytest.approx(0.3, abs=1e-12)
        assert std_len == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.unit
    def test_estimate_segment_length_3d_pythagorean(self) -> None:
        est = ParameterEstimator("dempster")
        prox = np.zeros((3, 3))
        dist = np.tile(np.array([3.0, 4.0, 0.0]), (3, 1))  # |.| = 5
        mean_len, _ = est.estimate_segment_length(prox, dist)
        assert mean_len == pytest.approx(5.0, abs=1e-12)

    @pytest.mark.unit
    def test_segment_mass_matches_dempster_fraction(self) -> None:
        est = ParameterEstimator("dempster")
        # Dempster upper_arm mass fraction = 0.028; body 70 kg -> 1.96 kg.
        params = est.estimate_segment_params("upper_arm", 0.3, 70.0)
        assert params.mass == pytest.approx(70.0 * 0.028, rel=1e-12)
        assert params.length == pytest.approx(0.3)
        assert params.com_position == pytest.approx(0.436)

    @pytest.mark.unit
    def test_thigh_mass_fraction(self) -> None:
        est = ParameterEstimator("dempster")
        params = est.estimate_segment_params("thigh", 0.4, 80.0)
        assert params.mass == pytest.approx(80.0 * 0.100, rel=1e-12)

    @pytest.mark.unit
    def test_inertia_positive_and_ordered(self) -> None:
        est = ParameterEstimator("dempster")
        params = est.estimate_segment_params("thigh", 0.4, 80.0)
        ixx, iyy, izz = params.inertia
        assert ixx > 0 and iyy > 0 and izz > 0
        assert ixx == pytest.approx(iyy)  # transverse axes equal
        # Long-axis inertia (thin segment) is far smaller than transverse.
        assert izz < ixx

    @pytest.mark.unit
    def test_unknown_segment_uses_defaults(self) -> None:
        est = ParameterEstimator("dempster")
        params = est.estimate_segment_params("tentacle", 0.5, 60.0)
        assert params.mass == pytest.approx(60.0 * 0.02, rel=1e-12)


# ---------------------------------------------------------------------------
# fit_parameters_to_kinematics: residual behaviour + empty-input contract
# ---------------------------------------------------------------------------
class TestFitParametersToKinematics:
    @pytest.mark.unit
    def test_empty_data_returns_failure(self) -> None:
        est = ParameterEstimator("dempster")
        result = est.fit_parameters_to_kinematics([], ["upper_arm"], 70.0)
        assert result.success is False
        assert result.rms_error == float("inf")

    @pytest.mark.unit
    def test_known_lengths_drive_residual_to_zero(self) -> None:
        est = ParameterEstimator("dempster")
        # Markers exactly 0.3 m apart along the chain; known length 0.3 ->
        # residual (mean_length - known) must be ~0.
        marker_frames = [np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]) for _ in range(4)]
        states = [
            KinematicState(timestamp=float(i), marker_positions=m)
            for i, m in enumerate(marker_frames)
        ]
        result = est.fit_parameters_to_kinematics(
            states, ["upper_arm"], 70.0, known_lengths={"upper_arm": 0.3}
        )
        assert result.success
        assert result.rms_error == pytest.approx(0.0, abs=1e-12)
        assert result.parameters["upper_arm_length"] == pytest.approx(0.3)

    @pytest.mark.unit
    def test_anthropometric_fallback_without_markers(self) -> None:
        est = ParameterEstimator("dempster")
        states = [KinematicState(timestamp=0.0, marker_positions=None)]
        result = est.fit_parameters_to_kinematics(
            states, ["upper_arm"], 70.0, known_lengths={"upper_arm": 0.3}
        )
        assert result.success
        assert "anthropometric" in result.message.lower()
        assert result.parameters["upper_arm_mass"] == pytest.approx(
            70.0 * 0.028, rel=1e-12
        )
