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


class TestMeshGeneratorFactory:
    """Test the MeshGenerator factory class."""

    def test_create_smplx(self) -> None:
        gen = MeshGenerator.create(MeshGeneratorBackend.SMPLX)
        assert isinstance(gen, SMPLXMeshGenerator)
        assert gen.backend_name == "smplx"

    def test_create_makehuman(self) -> None:
        gen = MeshGenerator.create(MeshGeneratorBackend.MAKEHUMAN)
        assert isinstance(gen, MakeHumanMeshGenerator)
        assert gen.backend_name == "makehuman"

    def test_create_from_string(self) -> None:
        gen = MeshGenerator.create("smplx")
        assert isinstance(gen, SMPLXMeshGenerator)

    def test_create_from_string_case_insensitive(self) -> None:
        gen = MeshGenerator.create("SMPLX")
        assert isinstance(gen, SMPLXMeshGenerator)

    def test_create_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            MeshGenerator.create(MeshGeneratorBackend.CUSTOM)

    def test_interface_compliance(self) -> None:
        """Verify both generators implement the full interface."""
        for cls in [SMPLXMeshGenerator, MakeHumanMeshGenerator]:
            gen = cls()
            assert isinstance(gen, MeshGeneratorInterface)
            assert isinstance(gen.backend_name, str)
            assert isinstance(gen.is_available, bool)
            assert isinstance(gen.get_supported_segments(), list)


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
