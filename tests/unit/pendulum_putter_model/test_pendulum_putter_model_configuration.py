"""Unit tests for PendulumPutterModel - TDD approach.

Tests the Perfy-style pendulum putter model based on Dave Pelz's design.
The model consists of a rigid stand with pendulum arms holding an
interchangeable putter club.

Tests follow the Pragmatic Programmer principles:
- Small, focused test functions
- Test one thing at a time
- Clear assertions with descriptive messages
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

# Skip entire module if model_generation.models.pendulum_putter is not available
# pendulum_putter submodule only exists at src/tools/model_generation in this repo.
pytest.importorskip(
    "model_generation.models.pendulum_putter",
    reason="model_generation.models.pendulum_putter package not available",
)


class TestPendulumPutterModelConfiguration:
    """Test model configuration and customization."""

    def test_can_set_arm_length(self) -> None:
        """Should be able to configure arm length."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        short_builder = PendulumPutterModelBuilder(arm_length_m=0.3)
        long_builder = PendulumPutterModelBuilder(arm_length_m=0.5)

        short_result = short_builder.build()
        long_result = long_builder.build()

        assert short_result.solver_status == "success", (
            "Assertion failed: short_result.solver_status == success"
        )
        assert long_result.solver_status == "success", (
            "Assertion failed: long_result.solver_status == success"
        )

        # Verify different configurations
        short_arm = short_result.get_link("pendulum_arm")
        long_arm = long_result.get_link("pendulum_arm")

        assert short_arm is not None, "Assertion failed: short_arm is not None"
        assert long_arm is not None, "Assertion failed: long_arm is not None"

    def test_can_set_shoulder_height(self) -> None:
        """Should be able to configure shoulder height."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(shoulder_height_m=1.0)
        result = builder.build()

        assert result.solver_status == "success", (
            "Assertion failed: result.solver_status == success"
        )

    def test_can_set_pendulum_damping(self) -> None:
        """Should be able to configure pendulum damping."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(damping=0.1)
        result = builder.build()

        joint = result.get_joint("pendulum_joint")
        assert joint.dynamics.damping == pytest.approx(0.1), (
            "Assertion failed: joint.dynamics.damping == pytest.approx(0.1)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
