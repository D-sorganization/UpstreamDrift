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


class TestSMPLXSegmentMesh:
    """Test the static _segment_mesh helper."""

    def test_segment_extracts_correct_vertices(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2]])
        faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

        seg_v, seg_f = SMPLXMeshGenerator._segment_mesh(verts, faces, 0, 4)
        # Should include only the first two faces (all vertices in [0,4))
        assert seg_v.shape[0] == 4
        assert seg_f.shape[0] == 2

    def test_empty_segment(self) -> None:
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])

        seg_v, seg_f = SMPLXMeshGenerator._segment_mesh(verts, faces, 0, 1)
        # Face uses vertices 0,1,2 but only vertex 0 is in [0,1)
        assert seg_f.shape[0] == 0


# ---------------------------------------------------------------------------
# MakeHuman Generator Tests  (See issue #979)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
