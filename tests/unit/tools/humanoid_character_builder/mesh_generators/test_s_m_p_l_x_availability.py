"""
Unit tests for SMPL-X and MakeHuman mesh generators.

Tests use mocked external dependencies (smplx, trimesh, subprocess) so that
the full pipeline logic can be validated without installing heavy optional
packages.

See issues #979 (MakeHuman) and #980 (SMPL-X).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from humanoid_character_builder.core.body_parameters import (
    BodyParameters,
    GenderModel,
)
from humanoid_character_builder.generators.mesh_generator import (
    GeneratedMeshResult,
    MakeHumanMeshGenerator,
    MeshGenerator,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
    SMPLXMeshGenerator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_params(**overrides: Any) -> BodyParameters:
    """Create default BodyParameters with optional overrides."""
    kwargs: dict[str, Any] = {
        "height_m": 1.80,
        "mass_kg": 80.0,
    }
    kwargs.update(overrides)
    return BodyParameters(**kwargs)


# ---------------------------------------------------------------------------
# SMPL-X Generator Tests  (See issue #980)
# ---------------------------------------------------------------------------


class TestSMPLXAvailability:
    """Test is_available and error paths."""

    def test_unavailable_when_smplx_missing(self) -> None:
        with patch(
            "humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE",
            False,
        ):
            gen = SMPLXMeshGenerator()
            assert gen.is_available is False

    def test_unavailable_when_model_dir_missing(self) -> None:
        with patch(
            "humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE",
            True,
        ):
            gen = SMPLXMeshGenerator(model_dir="/nonexistent/path")
            assert gen.is_available is False

    def test_returns_error_result_when_smplx_missing(self) -> None:
        with patch(
            "humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE",
            False,
        ):
            gen = SMPLXMeshGenerator()
            result = gen.generate(_default_params(), Path("/tmp/out"))
            assert result.solver_status != "success"
            assert "smplx" in result.error_message.lower()

    def test_returns_error_when_trimesh_missing(self) -> None:
        with (
            patch(
                "humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE",
                True,
            ),
            patch(
                "humanoid_character_builder.generators.mesh_generator.TRIMESH_AVAILABLE",
                False,
            ),
        ):
            gen = SMPLXMeshGenerator(model_dir="/nonexistent")
            result = gen.generate(_default_params(), Path("/tmp/out"))
            assert result.solver_status != "success"
            assert "trimesh" in result.error_message.lower()


# ---------------------------------------------------------------------------
# MakeHuman Generator Tests  (See issue #979)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
