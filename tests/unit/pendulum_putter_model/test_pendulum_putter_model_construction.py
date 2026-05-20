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


class TestPendulumPutterModelConstruction:
    """Test model construction and structure."""

    def test_model_builds_successfully(self) -> None:
        """Model should build without errors."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        assert (
            result.solver_status == "success"
        ), f"Build failed: {result.error_message}"
        assert (
            result.urdf_xml is not None
        ), "Assertion failed: result.urdf_xml is not None"

    def test_model_has_correct_link_count(self) -> None:
        """Model should have expected number of links."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Expected links: world, base, vertical_post, shoulder_mount,
        # pendulum_arm, club_mount, plus club links (grip, shaft, head)
        assert len(result.links) >= 6, "Should have at least 6 links"

    def test_model_has_single_dof_pendulum_joint(self) -> None:
        """Model should have exactly 1 DOF for pendulum motion."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Count revolute joints (DOF contributors)
        revolute_joints = [j for j in result.joints if j.joint_type.value == "revolute"]

        assert len(revolute_joints) == 1, "Should have exactly 1 revolute joint"
        assert result.get_total_dof() == 1, "Total DOF should be 1"

    def test_pendulum_joint_rotates_about_y_axis(self) -> None:
        """Pendulum joint should rotate about Y-axis for X-Z plane swing."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        pendulum_joint = result.get_joint("pendulum_joint")
        assert pendulum_joint is not None, "Should have pendulum_joint"

        # Y-axis rotation
        assert pendulum_joint.axis == (0, 1, 0), "Should rotate about Y-axis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
