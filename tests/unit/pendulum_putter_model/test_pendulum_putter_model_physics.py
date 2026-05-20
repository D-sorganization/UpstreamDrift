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

import math
from pathlib import Path
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


class TestPendulumPutterModelPhysics:
    """Test physical properties of the model."""

    def test_model_has_reasonable_total_mass(self) -> None:
        """Total mass should be reasonable for a putting robot."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        total_mass = result.get_total_mass()

        # Perfy-style robot: ~5-15 kg total
        assert 1.0 < total_mass < 30.0, f"Mass {total_mass} kg seems unreasonable"

    def test_base_is_heaviest_component(self) -> None:
        """Base should be heavy for stability."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        base_link = result.get_link("base_link")
        assert base_link is not None, "Assertion failed: base_link is not None"

        # Base should be at least 30% of total mass for stability
        base_mass = base_link.inertia.mass
        total_mass = result.get_total_mass()
        assert base_mass > 0.3 * total_mass, "Base should be heavy for stability"

    def test_all_inertias_are_physically_valid(self) -> None:
        """All inertia tensors should be positive definite."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        for link in result.links:
            if link.inertia.mass > 1e-6:  # Skip negligible mass links
                assert (
                    link.inertia.is_positive_definite()
                ), f"Link {link.name} has non-positive-definite inertia"

    def test_pendulum_joint_has_appropriate_limits(self) -> None:
        """Pendulum joint should have reasonable angle limits."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        joint = result.get_joint("pendulum_joint")
        assert joint is not None, "Assertion failed: joint is not None"
        assert joint.limits is not None, "Assertion failed: joint.limits is not None"

        # Putting stroke: typically ±30-45 degrees max
        assert joint.limits.lower >= -math.pi / 2, "Lower limit too extreme"
        assert joint.limits.upper <= math.pi / 2, "Upper limit too extreme"

    def test_pendulum_joint_has_low_damping(self) -> None:
        """Pendulum should have low damping for free swing."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        joint = result.get_joint("pendulum_joint")
        assert joint is not None, "Assertion failed: joint is not None"

        # Low damping for pendulum behavior
        assert joint.dynamics.damping < 0.5, "Damping too high for pendulum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
