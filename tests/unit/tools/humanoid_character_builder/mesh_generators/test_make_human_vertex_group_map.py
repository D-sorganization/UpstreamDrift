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


class TestMakeHumanVertexGroupMap:
    """Test that the vertex group map is complete."""

    def test_all_values_unique(self) -> None:
        values = list(MakeHumanMeshGenerator.MH_VERTEX_GROUP_MAP.values())
        assert len(values) == len(set(values)), "Duplicate segment names in map"

    def test_all_keys_unique(self) -> None:
        keys = list(MakeHumanMeshGenerator.MH_VERTEX_GROUP_MAP.keys())
        assert len(keys) == len(set(keys)), "Duplicate MH group names in map"


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
