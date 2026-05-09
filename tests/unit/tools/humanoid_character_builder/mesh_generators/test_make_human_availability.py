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


class TestMakeHumanAvailability:
    """Test is_available for MakeHuman."""

    def test_unavailable_when_not_installed(self) -> None:
        gen = MakeHumanMeshGenerator(makehuman_path="/nonexistent/makehuman")
        assert gen.is_available is False

    def test_available_when_path_exists(self, tmp_path: Path) -> None:
        mh_dir = tmp_path / "makehuman"
        mh_dir.mkdir()
        gen = MakeHumanMeshGenerator(makehuman_path=mh_dir)
        assert gen.is_available is True

    def test_returns_error_when_unavailable(self, tmp_path: Path) -> None:
        gen = MakeHumanMeshGenerator(makehuman_path="/nonexistent")
        result = gen.generate(_default_params(), tmp_path / "out")
        assert result.solver_status != "success"
        assert "not found" in result.error_message.lower()


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
