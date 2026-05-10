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


class TestInterchangeableClub:
    """Test club interchangeability feature."""

    def test_default_club_is_attached(self) -> None:
        """Model should have a default putter attached."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Should have club-related links
        club_links = [
            link
            for link in result.links
            if "club" in link.name.lower()
            or "putter" in link.name.lower()
            or "grip" in link.name.lower()
            or "shaft" in link.name.lower()
            or "head" in link.name.lower()
        ]
        assert len(club_links) >= 1, "Should have club links attached"

    def test_can_build_without_club(self) -> None:
        """Should be able to build model without a club."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(include_club=False)
        result = builder.build()

        assert (
            result.solver_status == "success"
        ), "Assertion failed: result.solver_status == success"

        # Should end with club_mount
        club_mount = result.get_link("club_mount")
        assert club_mount is not None, "Assertion failed: club_mount is not None"

    def test_can_attach_custom_club(self) -> None:
        """Should be able to attach a custom club configuration."""
        from model_generation.models.pendulum_putter import (
            ClubConfig,
            PendulumPutterModelBuilder,
        )

        custom_club = ClubConfig(
            grip_length_m=0.25,
            shaft_length_m=0.85,
            head_mass_kg=0.35,
        )

        builder = PendulumPutterModelBuilder(club_config=custom_club)
        result = builder.build()

        assert (
            result.solver_status == "success"
        ), "Assertion failed: result.solver_status == success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
