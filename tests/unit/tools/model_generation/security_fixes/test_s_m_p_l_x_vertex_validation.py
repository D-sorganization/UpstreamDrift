"""Tests for security hardening: REST API auth/CORS/rate-limit, cache URL
validation, and SMPL-X vertex range validation.

Covers GitHub issues #1695, #1691, #1700.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. REST API security tests (issue #1695)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. URL validation and path traversal tests (issue #1700)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. SMPL-X vertex range validation tests (issue #1691)
# ---------------------------------------------------------------------------


class TestSMPLXVertexValidation:
    """SMPL-X hardcoded vertex range validation."""

    def test_expected_vertex_count_constant_exists(self) -> None:
        """SMPLX_EXPECTED_VERTEX_COUNT should be defined."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        assert hasattr(SMPLXMeshGenerator, "SMPLX_EXPECTED_VERTEX_COUNT")
        assert SMPLXMeshGenerator.SMPLX_EXPECTED_VERTEX_COUNT == 10475

    def test_vertex_ranges_within_expected_count(self) -> None:
        """All vertex ranges should be within [0, SMPLX_EXPECTED_VERTEX_COUNT)."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        expected = SMPLXMeshGenerator.SMPLX_EXPECTED_VERTEX_COUNT
        for name, (
            start,
            end,
        ) in SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES.items():
            assert 0 <= start < expected, f"{name}: start {start} out of range"
            assert 0 < end <= expected, f"{name}: end {end} out of range"
            assert start < end, f"{name}: start {start} >= end {end}"

    def test_validate_vertex_ranges_method_exists(self) -> None:
        """validate_vertex_ranges class method should exist."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        assert hasattr(SMPLXMeshGenerator, "validate_vertex_ranges")

    def test_validate_vertex_ranges_passes_for_matching_count(self) -> None:
        """validate_vertex_ranges returns True when vertex count matches."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        result = SMPLXMeshGenerator.validate_vertex_ranges(10475)
        assert result is True

    def test_validate_vertex_ranges_warns_for_mismatched_count(self) -> None:
        """validate_vertex_ranges returns False for wrong vertex count."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        result = SMPLXMeshGenerator.validate_vertex_ranges(5000)
        assert result is False

    def test_load_segmentation_from_file_method_exists(self) -> None:
        """load_part_segmentation classmethod should exist."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        assert hasattr(SMPLXMeshGenerator, "load_part_segmentation")

    def test_load_segmentation_falls_back_to_hardcoded(self) -> None:
        """When no model file is available, should fall back to hardcoded
        ranges and log a warning."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        # Call with a non-existent path
        result = SMPLXMeshGenerator.load_part_segmentation(Path("/nonexistent/path"))
        # Should return the hardcoded ranges
        assert isinstance(result, dict)
        assert len(result) > 0
        # The returned dict should match SMPLX_SEGMENT_VERTEX_RANGES
        assert result == SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES

    def test_load_segmentation_logs_warning_on_fallback(self) -> None:
        """Falling back to hardcoded ranges should produce a warning log."""
        from humanoid_character_builder.generators.mesh_generator import (
            SMPLXMeshGenerator,
        )

        with patch(
            "humanoid_character_builder.generators.mesh_generator.logger"
        ) as mock_logger:
            SMPLXMeshGenerator.load_part_segmentation(Path("/nonexistent/path"))
            mock_logger.warning.assert_called()
            # The warning should mention fallback or hardcoded
            call_args = str(mock_logger.warning.call_args)
            assert (
                "hardcoded" in call_args.lower()
                or "fallback" in call_args.lower()
                or "fall" in call_args.lower()
            )
