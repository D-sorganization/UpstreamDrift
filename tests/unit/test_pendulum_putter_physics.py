"""Physics validation tests for PendulumPutterModel.

Tests that the generated URDF model meets physics engine requirements
and has correct physical properties.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# Skip entire module if model_generation.models.pendulum_putter is not available
pytest.importorskip(
    "model_generation.models.pendulum_putter",
    reason="model_generation.models.pendulum_putter package not available",
)


# Check if mujoco is available without causing module-level skip
def _mujoco_available() -> bool:
    """Check if MuJoCo is available."""
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


HAS_MUJOCO = _mujoco_available()


class TestModelValidationForEngines:
    """Test that model meets requirements for physics engine loading."""

    @pytest.fixture
    def model_result(self) -> Any:
        """Build model result for testing."""
        from model_generation.models.pendulum_putter import PendulumPutterModelBuilder

        builder = PendulumPutterModelBuilder()
        return builder.build()

    def test_all_links_have_inertial(self, model_result) -> None:
        """All links (except world) should have valid inertial properties."""
        for link in model_result.links:
            if link.name == "world":
                continue

            assert link.inertia is not None, f"Link {link.name} missing inertia"
            assert link.inertia.mass > 0, f"Link {link.name} has zero/negative mass"
            assert link.inertia.is_positive_definite(), (
                f"Link {link.name} has invalid inertia tensor"
            )

    def test_all_joints_have_valid_axes(self, model_result) -> None:
        """All joints should have valid, normalized axes."""
        for joint in model_result.joints:
            if joint.joint_type.value == "fixed":
                continue

            axis = np.array(joint.axis)
            norm = np.linalg.norm(axis)
            assert abs(norm - 1.0) < 1e-6, (
                f"Joint {joint.name} axis not normalized: {joint.axis}"
            )

    def test_joint_limits_are_consistent(self, model_result) -> None:
        """Joint limits should be valid (lower < upper)."""
        for joint in model_result.joints:
            if joint.limits is None:
                continue

            assert joint.limits.lower < joint.limits.upper, (
                f"Joint {joint.name} has invalid limits: "
                f"lower={joint.limits.lower}, upper={joint.limits.upper}"
            )

    def test_model_has_valid_tree_structure(self, model_result) -> None:
        """Model should have valid parent-child relationships."""
        link_names = {link.name for link in model_result.links}

        for joint in model_result.joints:
            assert joint.parent in link_names, (
                f"Joint {joint.name} has missing parent: {joint.parent}"
            )
            assert joint.child in link_names, (
                f"Joint {joint.name} has missing child: {joint.child}"
            )


class TestPendulumPhysicsProperties:
    """Test physics properties of the pendulum model."""

    def test_natural_frequency_estimation(self) -> None:
        """Natural frequency should be calculable and reasonable."""
        from model_generation.models.pendulum_putter import PendulumPutterModelBuilder

        builder = PendulumPutterModelBuilder(arm_length_m=0.4, include_club=False)
        result = builder.build()

        freq_hz = result.metadata.get("natural_frequency_approx_hz", 0)

        # For arm_length=0.4m: f ≈ (1/2π) * sqrt(9.81/0.4) ≈ 0.79 Hz
        assert 0.3 < freq_hz < 2.0, f"Natural frequency {freq_hz} Hz seems unreasonable"

    def test_pendulum_period_depends_on_arm_length(self) -> None:
        """Longer arm should have longer period (lower frequency)."""
        from model_generation.models.pendulum_putter import PendulumPutterModelBuilder

        short_builder = PendulumPutterModelBuilder(arm_length_m=0.3, include_club=False)
        long_builder = PendulumPutterModelBuilder(arm_length_m=0.6, include_club=False)

        short_result = short_builder.build()
        long_result = long_builder.build()

        short_freq = short_result.metadata.get("natural_frequency_approx_hz", 0)
        long_freq = long_result.metadata.get("natural_frequency_approx_hz", 0)

        assert long_freq < short_freq, (
            f"Longer arm should have lower frequency: "
            f"short={short_freq} Hz, long={long_freq} Hz"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
