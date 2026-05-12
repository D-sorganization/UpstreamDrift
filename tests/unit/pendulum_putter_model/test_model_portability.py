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


class TestModelPortability:
    """Test that model can be moved around environments."""

    def test_world_link_is_root(self) -> None:
        """World link should be the root for attachment."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        root = result.get_root_link()
        assert root is not None, "Assertion failed: root is not None"
        assert root.name == "world", "Assertion failed: root.name == world"

    def test_base_attached_to_world_via_fixed_joint(self) -> None:
        """Base should be attached to world via fixed joint for positioning."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Find joint connecting world to base
        world_to_base = result.get_joint("world_to_base")
        assert world_to_base is not None, "Assertion failed: world_to_base is not None"
        assert (
            world_to_base.joint_type.value == "fixed"
        ), "Assertion failed: world_to_base.joint_type.value == fixed"
        assert (
            world_to_base.parent == "world"
        ), "Assertion failed: world_to_base.parent == world"
        assert (
            world_to_base.child == "base_link"
        ), "Assertion failed: world_to_base.child == base_link"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
