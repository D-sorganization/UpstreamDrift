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


class TestValidation:
    """Test model validation."""

    def test_validation_passes_for_default_model(self) -> None:
        """Default model should pass all validations."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        assert result.validation is not None, (
            "Assertion failed: result.validation is not None"
        )
        assert result.validation.is_valid, (
            f"Validation failed: {result.validation.get_error_messages()}"
        )

    def test_validation_catches_invalid_parameters(self) -> None:
        """Should catch invalid configuration parameters."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        # Negative arm length should raise
        with pytest.raises(ValueError, match="arm_length"):
            PendulumPutterModelBuilder(arm_length_m=-0.5)

        # Negative shoulder height should raise
        with pytest.raises(ValueError, match="shoulder_height"):
            PendulumPutterModelBuilder(shoulder_height_m=-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
