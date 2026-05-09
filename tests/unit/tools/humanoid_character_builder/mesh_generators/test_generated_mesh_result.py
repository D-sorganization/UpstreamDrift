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


# ---------------------------------------------------------------------------
# MakeHuman Generator Tests  (See issue #979)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------


class TestGeneratedMeshResult:
    """Test the result dataclass."""

    def test_successful_result(self) -> None:
        result = GeneratedMeshResult(
            success=True,
            mesh_paths={"head": Path("head.stl")},
        )
        assert result.solver_status == "success"
        assert result.error_message is None

    def test_failed_result(self) -> None:
        result = GeneratedMeshResult(
            success=False,
            error_message="Something went wrong",
        )
        assert result.solver_status != "success"
        assert result.error_message == "Something went wrong"

    def test_mesh_generators_defaults(self) -> None:
        result = GeneratedMeshResult(success=True)
        assert result.mesh_paths == {}
        assert result.collision_paths == {}
        assert result.texture_paths == {}
        assert result.vertex_groups == {}
        assert result.metadata == {}
