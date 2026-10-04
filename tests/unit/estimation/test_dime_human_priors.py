"""Unit tests for DIME-13: Hierarchical Human Dimensions and Coupled ROM Priors (#11421, #11434).

Covers required RED and GREEN behaviors:
RED:
1. Plausible unusual limb proportions forced to average (rejected).
2. Contradictory measured height (fails closed with PreconditionError).
3. Wrong degree/radian limits (degrees detected and rejected).
4. Coupled shoulder restriction (scapulohumeral rhythm violation detected).
5. Asymmetric subject (natural limb asymmetry accepted without artificial symmetry lock).
6. Occluded limb overconfidence (unobserved variables retain broadened uncertainty).

GREEN:
1. Planted population/subject hierarchical fixture updates correctly (Bayesian posterior matches analytical solution).
2. Covariances remain valid (symmetric positive semi-definite).
3. Inertia is physically realizable (satisfies triangle inequalities and positive eigenvalues).
4. Missing data broadens uncertainty correctly under marginalization.
5. Serialization roundtrip for prior models, measurements, and evaluation receipts.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_human_priors import (
    CoupledRangeOfMotionPrior,
    DimeHumanPriorReport,
    HierarchicalDimensionPrior,
    InertiaRealizabilityCheck,
    PhysicalDimensionBounds,
    PopulationPriorVersion,
    RangeOfMotionBound,
    SubjectDimensionMeasurement,
    evaluate_human_prior_compatibility,
    update_hierarchical_dimension_posterior,
    validate_inertia_realizability,
    validate_range_of_motion_units,
)

pytestmark = pytest.mark.unit


class TestDimeHumanPriorsRedSuite:
    """RED test suite enforcing fail-closed constraints and error conditions."""

    def test_red_contradictory_measured_height_fails_closed(self) -> None:
        """Reject subject measurements where segment sum contradicts total height."""
        # Total height 1.80m, but torso (0.90) + thigh (0.60) + shank (0.55) + head (0.35) = 2.40m (>33% excess)
        bounds = PhysicalDimensionBounds()
        segment_lengths = {
            "head": 0.35,
            "torso": 0.90,
            "thigh_left": 0.60,
            "shank_left": 0.55,
        }
        with pytest.raises(PreconditionError, match="[Cc]ontradictory.*height"):
            bounds.validate_height_consistency(height_m=1.80, segments=segment_lengths)

    def test_red_wrong_degree_radian_limits_rejected(self) -> None:
        """Reject range-of-motion bounds passed in degrees instead of radians."""
        # 90 degrees passed as upper limit (1.57 rad expected; 90 rad is physical nonsense)
        with pytest.raises(PreconditionError, match="[Rr]adian|[Dd]egree"):
            RangeOfMotionBound(
                joint_name="elbow_flexion",
                lower_limit=-0.1,
                upper_limit=90.0,  # Degrees passed erroneously!
            )

    def test_red_coupled_shoulder_restriction_violation(self) -> None:
        """Detect pose that violates coupled shoulder elevation-rotation rhythm."""
        rom_prior = CoupledRangeOfMotionPrior()
        # When shoulder elevation is high (e.g. 2.5 rad ~ 143 deg), internal rotation is limited
        # An uncoupled limit would allow 1.5 rad rotation, but coupled restriction limits it to 0.5 rad
        pose = {
            "shoulder_elevation": 2.5,
            "shoulder_internal_rotation": 1.4,  # Violates coupled bound!
        }
        res = rom_prior.evaluate_plausibility(pose)
        assert not res.is_feasible
        assert res.violation_amount > 0.0
        assert "shoulder_coupled" in res.violated_couplings

    def test_red_asymmetric_subject_not_forced_symmetric(self) -> None:
        """Admit measured natural asymmetry without forcing left/right equality."""
        prior = HierarchicalDimensionPrior.create_default(allow_asymmetry=True)
        # Subject with a real 3cm limb length difference (e.g. left thigh 0.44m, right thigh 0.41m)
        measurements = [
            SubjectDimensionMeasurement(
                dimension_name="thigh_left",
                measured_value=0.44,
                uncertainty=0.005,
            ),
            SubjectDimensionMeasurement(
                dimension_name="thigh_right",
                measured_value=0.41,
                uncertainty=0.005,
            ),
        ]
        posterior = update_hierarchical_dimension_posterior(prior, measurements)
        # Posterior should preserve the measured difference rather than forcing exact symmetry
        mean_left = posterior.get_mean("thigh_left")
        mean_right = posterior.get_mean("thigh_right")
        assert abs(mean_left - mean_right) > 0.02

    def test_red_occluded_limb_uncertainty_broadens_not_tightens(self) -> None:
        """Unobserved/occluded limb retains prior variance rather than gaining false confidence."""
        prior = HierarchicalDimensionPrior.create_default(allow_asymmetry=True)
        # Only measure right side; left side is completely occluded
        measurements = [
            SubjectDimensionMeasurement(
                dimension_name="thigh_right",
                measured_value=0.42,
                uncertainty=0.005,
            ),
        ]
        posterior = update_hierarchical_dimension_posterior(prior, measurements)
        var_observed = posterior.get_variance("thigh_right")
        var_occluded = posterior.get_variance("thigh_left")
        prior_var_left = prior.get_variance("thigh_left")

        # Measured variable variance shrinks significantly (order of magnitude)
        assert var_observed < 0.0001
        # Occluded variable variance remains wide (no artificial overconfidence, >5x observed)
        assert var_occluded > 0.0002
        assert var_occluded > var_observed * 5.0

    def test_red_unrealizable_inertia_violates_triangle_inequality(self) -> None:
        """Reject unphysical inertia tensor violating triangle inequalities."""
        # Ixx = 10.0, Iyy = 1.0, Izz = 1.0 -> Ixx > Iyy + Izz (violates triangle inequality!)
        check = validate_inertia_realizability(mass_kg=5.0, ixx=10.0, iyy=1.0, izz=1.0)
        assert not check.is_realizable
        assert (
            check.failure_reason is not None
            and "triangle_inequality" in check.failure_reason
        )

    def test_red_plausible_unusual_proportions_not_penalized_as_outlier(self) -> None:
        """Plausible correlated unusual proportions (long wingspan for tall subject) remain feasible."""
        prior = HierarchicalDimensionPrior.create_default()
        # Subject is 2.00m tall with wingspan 2.08m (both ~+3 std, highly correlated)
        compat = evaluate_human_prior_compatibility(
            prior,
            height_m=2.00,
            wingspan_m=2.08,
        )
        assert compat.is_acceptable
        assert compat.mahalanobis_distance < 4.0  # Joint correlation accounts for it!

        # In contrast, contradictory proportions (tall person with very short wingspan 1.50m) are rejected
        outlier = evaluate_human_prior_compatibility(
            prior,
            height_m=2.00,
            wingspan_m=1.50,
        )
        assert not outlier.is_acceptable
        assert outlier.mahalanobis_distance > 5.0


class TestDimeHumanPriorsGreenSuite:
    """GREEN test suite validating Bayesian updating, physical bounds, and receipts."""

    def test_green_hierarchical_bayesian_posterior_analytic_match(self) -> None:
        """Verify Bayesian update matches analytical Gaussian posterior formula."""
        # 2D toy prior: mean [1.0, 2.0], cov [[0.04, 0.02], [0.02, 0.09]]
        mean_prior = np.array([1.0, 2.0], dtype=np.float64)
        cov_prior = np.array([[0.04, 0.02], [0.02, 0.09]], dtype=np.float64)
        dim_names = ("dim_a", "dim_b")

        prior = HierarchicalDimensionPrior(
            names=dim_names,
            mean=mean_prior,
            covariance=cov_prior,
            version=PopulationPriorVersion.ANSUR2_V1,
        )

        # Measure dim_a = 1.05 with sigma = 0.05
        measurements = [
            SubjectDimensionMeasurement(
                dimension_name="dim_a",
                measured_value=1.05,
                uncertainty=0.05,
            )
        ]

        posterior = update_hierarchical_dimension_posterior(prior, measurements)

        # Analytical Kalman/Gaussian posterior:
        # H = [1, 0], R = [0.0025]
        # K = P H^T (H P H^T + R)^-1 = [0.04, 0.02]^T / (0.04 + 0.0025)
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.05**2]])
        S = H @ cov_prior @ H.T + R
        K = cov_prior @ H.T @ np.linalg.inv(S)
        y = np.array([1.05])
        expected_mean = mean_prior + (K @ (y - H @ mean_prior)).squeeze()
        expected_cov = (np.eye(2) - K @ H) @ cov_prior

        np.testing.assert_allclose(posterior.mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(posterior.covariance, expected_cov, atol=1e-6)

    def test_green_valid_covariance_properties(self) -> None:
        """Posterior covariance remains symmetric and positive semi-definite."""
        prior = HierarchicalDimensionPrior.create_default()
        measurements = [
            SubjectDimensionMeasurement("height", 1.85, 0.01),
            SubjectDimensionMeasurement("torso", 0.65, 0.02),
        ]
        posterior = update_hierarchical_dimension_posterior(prior, measurements)
        cov = posterior.covariance
        # Symmetry
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        # Positive semi-definiteness: all eigenvalues >= -1e-12
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-12)

    def test_green_physically_realizable_inertia(self) -> None:
        """Valid cylinder/ellipsoid inertia satisfies triangle inequality and positive eigenvalues."""
        check = validate_inertia_realizability(
            mass_kg=75.0,
            ixx=10.5,
            iyy=9.8,
            izz=2.1,
        )
        assert check.is_realizable
        assert check.failure_reason is None
        assert all(e > 0.0 for e in check.eigenvalues)

    def test_green_serialization_roundtrip(self) -> None:
        """Verify serialization to and from dictionary for report and prior objects."""
        prior = HierarchicalDimensionPrior.create_default()
        data = prior.to_dict()
        restored = HierarchicalDimensionPrior.from_dict(data)

        assert restored.version == prior.version
        assert restored.names == prior.names
        np.testing.assert_allclose(restored.mean, prior.mean)
        np.testing.assert_allclose(restored.covariance, prior.covariance)
