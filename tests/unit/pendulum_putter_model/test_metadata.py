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


class TestMetadata:
    """Test model metadata for documentation and traceability."""

    def test_metadata_includes_model_name(self) -> None:
        """Metadata should include model identification."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        assert "robot_name" in result.metadata, (
            "Assertion failed: robot_name in result.metadata"
        )
        assert result.metadata["robot_name"] == "pendulum_putter", (
            "Assertion failed: result.metadata[robot_name] == pendulum_putter"
        )

    def test_metadata_includes_configuration(self) -> None:
        """Metadata should include configuration parameters."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(
            arm_length_m=0.35,
            shoulder_height_m=0.9,
        )
        result = builder.build()

        assert "arm_length_m" in result.metadata, (
            "Assertion failed: arm_length_m in result.metadata"
        )
        assert "shoulder_height_m" in result.metadata, (
            "Assertion failed: shoulder_height_m in result.metadata"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
