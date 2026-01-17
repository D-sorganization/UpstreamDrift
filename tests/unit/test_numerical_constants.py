import sys
from pathlib import Path

# Ensure repo root is in path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from shared.python.numerical_constants import (
    CONDITION_NUMBER_CRITICAL_THRESHOLD,
    CONDITION_NUMBER_WARNING_THRESHOLD,
    EPSILON_FINITE_DIFF_JACOBIAN,
    EPSILON_MASS_MATRIX_REGULARIZATION,
    EPSILON_SINGULARITY_DETECTION,
    GRAVITY_STANDARD,
    HUMAN_BODY_MASS_PLAUSIBLE_RANGE,
    SEGMENT_LENGTH_TO_HEIGHT_RATIO_PLAUSIBLE,
    TOLERANCE_ENERGY_CONSERVATION,
    TOLERANCE_WORK_ENERGY_MISMATCH,
)


class TestNumericalConstants:
    """Regression tests for numerical constants."""

    def test_finite_difference_epsilon(self) -> None:
        """Verify finite difference step size."""
        # Must be small but not too small (sqrt(epsilon_machine))
        assert EPSILON_FINITE_DIFF_JACOBIAN == 1e-6
        assert 1e-8 <= EPSILON_FINITE_DIFF_JACOBIAN <= 1e-4

    def test_singularity_epsilon(self) -> None:
        """Verify singularity detection threshold."""
        assert EPSILON_SINGULARITY_DETECTION == 1e-10
        assert EPSILON_SINGULARITY_DETECTION > 0

    def test_mass_matrix_regularization(self) -> None:
        """Verify mass matrix regularization term."""
        assert EPSILON_MASS_MATRIX_REGULARIZATION == 1e-10

    def test_energy_tolerances(self) -> None:
        """Verify energy conservation tolerances."""
        assert TOLERANCE_ENERGY_CONSERVATION == 1e-6
        assert TOLERANCE_WORK_ENERGY_MISMATCH == 0.05
        # Work-energy mismatch is looser than conservation
        assert TOLERANCE_WORK_ENERGY_MISMATCH > TOLERANCE_ENERGY_CONSERVATION

    def test_condition_number_thresholds(self) -> None:
        """Verify condition number thresholds."""
        assert CONDITION_NUMBER_WARNING_THRESHOLD == 1e6
        assert CONDITION_NUMBER_CRITICAL_THRESHOLD == 1e10
        assert CONDITION_NUMBER_CRITICAL_THRESHOLD > CONDITION_NUMBER_WARNING_THRESHOLD

    def test_physical_constants(self) -> None:
        """Verify standard physical constants."""
        # Gravity should be roughly 9.81
        assert 9.78 <= GRAVITY_STANDARD <= 9.83

        # Body mass range
        min_mass, max_mass = HUMAN_BODY_MASS_PLAUSIBLE_RANGE
        assert min_mass == 40.0
        assert max_mass == 200.0
        assert min_mass < max_mass

    def test_segment_ratios(self) -> None:
        """Verify segment length ratios."""
        ratios = SEGMENT_LENGTH_TO_HEIGHT_RATIO_PLAUSIBLE

        required_segments = ["upper_arm", "forearm", "hand", "thigh", "shank", "foot"]
        for segment in required_segments:
            assert segment in ratios
            min_ratio, max_ratio = ratios[segment]
            assert 0.0 < min_ratio < max_ratio < 1.0

        # Thigh should be longer than foot
        assert ratios["thigh"][0] > ratios["foot"][1]
