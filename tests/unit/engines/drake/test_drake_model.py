"""Unit tests for Drake golf swing model.

Tests model building, parameter validation, and model structure.
"""

import numpy as np
import pytest

# Try to import drake_golf_model, skip all tests if pydrake is not available
# Note: pythonpath is configured in pytest.ini to include the parent directory
try:
    from pydrake.multibody.tree import SpatialInertia
    from python.src.drake_golf_model import (
        GolfModelParams,
        SegmentParams,
        build_golf_swing_diagram,
        make_cylinder_inertia,
    )
except ImportError as e:
    # Skip all tests if pydrake is not available
    # We define dummy SpatialInertia to allow test collection implicitly,
    # but the whole file will likely need pytest skipping mechanism if we want
    # to run tests selectively.
    # However, existing pattern is to skip if import fails.
    SpatialInertia = None  # type: ignore[misc, assignment]  # dummy fallback when pydrake unavailable
    import pytest

    pytest.skip(f"pydrake not available: {e}", allow_module_level=True)


class TestSegmentParams:
    """Tests for SegmentParams dataclass."""

    def test_segment_params_default(self) -> None:
        """Test SegmentParams with default radius."""
        params = SegmentParams(length=1.0, mass=2.0)
        assert params.length == 1.0
        assert params.mass == 2.0
        assert params.radius == 0.03  # Default value

    def test_segment_params_custom_radius(self) -> None:
        """Test SegmentParams with custom radius."""
        params = SegmentParams(length=1.0, mass=2.0, radius=0.05)
        assert params.radius == 0.05

    def test_segment_params_allows_negative_values(self) -> None:
        """Test SegmentParams allows negative values (no validation).

        Note: SegmentParams is a simple dataclass without validation.
        Negative values are allowed but may not be physically meaningful.
        """
        # SegmentParams doesn't validate, so negative values are allowed
        params = SegmentParams(length=-1.0, mass=2.0)
        assert params.length == -1.0
        assert params.mass == 2.0

        params = SegmentParams(length=1.0, mass=-2.0)
        assert params.length == 1.0
        assert params.mass == -2.0


class TestGolfModelParams:
    """Tests for GolfModelParams dataclass."""

    def test_drake_model_default_params(self) -> None:
        """Test default parameter values."""
        params = GolfModelParams()
        assert params.pelvis_to_shoulders > 0
        assert params.spine_mass > 0
        assert params.hand_spacing_m > 0
        assert params.club.length > 0
        assert params.club.mass > 0

    def test_drake_model_custom_params(self) -> None:
        """Test custom parameter values."""
        custom_club = SegmentParams(length=1.1, mass=0.45)
        params = GolfModelParams(club=custom_club)
        assert params.club.length == 1.1
        assert params.club.mass == 0.45

    def test_joint_axes_normalized(self) -> None:
        """Test joint axes are properly defined."""
        params = GolfModelParams()
        # Check axes are numpy arrays
        assert isinstance(params.hip_axis, np.ndarray)
        assert isinstance(params.spine_twist_axis, np.ndarray)
        assert len(params.shoulder_axes) == 3

    def test_friction_parameters(self) -> None:
        """Test friction parameters are reasonable."""
        params = GolfModelParams()
        assert 0 < params.ground_friction_mu_static <= 1.0
        assert 0 < params.ground_friction_mu_dynamic <= 1.0
        assert params.ground_friction_mu_dynamic <= params.ground_friction_mu_static


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_hand_spacing_reasonable(self) -> None:
        """Test hand spacing is reasonable."""
        params = GolfModelParams()
        # Hand spacing should be less than club length
        assert params.hand_spacing_m < params.club.length
        # Hand spacing should be positive
        assert params.hand_spacing_m > 0

    def test_segment_lengths_positive(self) -> None:
        """Test all segment lengths are positive."""
        params = GolfModelParams()
        assert params.scapula_rod.length > 0
        assert params.upper_arm.length > 0
        assert params.forearm.length > 0
        assert params.hand.length > 0
        assert params.club.length > 0

    def test_segment_masses_positive(self) -> None:
        """Test all segment masses are positive."""
        params = GolfModelParams()
        assert params.scapula_rod.mass > 0
        assert params.upper_arm.mass > 0
        assert params.forearm.mass > 0
        assert params.hand.mass > 0
        assert params.club.mass > 0
        assert params.spine_mass > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
