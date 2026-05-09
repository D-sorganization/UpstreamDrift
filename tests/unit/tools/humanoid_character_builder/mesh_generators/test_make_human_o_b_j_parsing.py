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


class TestMakeHumanOBJParsing:
    """Test the OBJ file parser."""

    def test_parse_simple_obj(self, tmp_path: Path) -> None:
        obj_content = textwrap.dedent("""\
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 0.0 1.0 0.0
            v 1.0 1.0 0.0
            f 1 2 3
            f 2 3 4
        """)
        obj_file = tmp_path / "test.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        vertices, faces = MakeHumanMeshGenerator._parse_obj_file(obj_file)
        assert vertices.shape == (4, 3)
        assert faces.shape == (2, 3)
        # OBJ is 1-indexed, so first face should be [0, 1, 2]
        assert faces[0].tolist() == [0, 1, 2]

    def test_parse_obj_with_normals_and_texcoords(self, tmp_path: Path) -> None:
        obj_content = textwrap.dedent("""\
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 0.0 1.0 0.0
            vn 0.0 0.0 1.0
            vt 0.0 0.0
            f 1/1/1 2/1/1 3/1/1
        """)
        obj_file = tmp_path / "test.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        vertices, faces = MakeHumanMeshGenerator._parse_obj_file(obj_file)
        assert vertices.shape == (3, 3)
        assert faces.shape == (1, 3)

    def test_parse_obj_quad_triangulation(self, tmp_path: Path) -> None:
        obj_content = textwrap.dedent("""\
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 1.0 1.0 0.0
            v 0.0 1.0 0.0
            f 1 2 3 4
        """)
        obj_file = tmp_path / "test.obj"
        obj_file.write_text(obj_content, encoding="utf-8")

        vertices, faces = MakeHumanMeshGenerator._parse_obj_file(obj_file)
        assert vertices.shape == (4, 3)
        # A quad should be split into 2 triangles
        assert faces.shape == (2, 3)


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
