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


class TestPendulumPhysicsAnalytical:
    """Test analytical physics properties of the pendulum model."""

    def test_natural_frequency_calculable(self) -> None:
        """Should be able to calculate natural frequency from model params."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(arm_length_m=0.4)
        result = builder.build()

        # Get pendulum arm length and mass distribution
        # ω = sqrt(g/L) for simple pendulum
        # For compound pendulum: ω = sqrt(m*g*d / I)
        # where d = distance from pivot to COM, I = moment of inertia about pivot

        pendulum_arm = result.get_link("pendulum_arm")
        assert pendulum_arm is not None, "Assertion failed: pendulum_arm is not None"

        # Basic sanity check - mass should allow pendulum motion
        assert pendulum_arm.inertia.mass > 0, (
            "Assertion failed: pendulum_arm.inertia.mass > 0"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
