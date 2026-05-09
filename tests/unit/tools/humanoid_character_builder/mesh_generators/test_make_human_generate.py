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


class TestMakeHumanGenerate:
    """Test the full MakeHuman generate pipeline with mocking."""

    @patch(
        "humanoid_character_builder.generators.mesh_generator.TRIMESH_AVAILABLE", True
    )
    @patch("humanoid_character_builder.generators.mesh_generator._trimesh_module")
    def test_generate_with_mocked_subprocess(
        self, mock_trimesh, tmp_path: Path
    ) -> None:
        """Test end-to-end generation with mocked MakeHuman subprocess."""
        mh_dir = tmp_path / "makehuman"
        mh_dir.mkdir()
        (mh_dir / "makehuman.py").touch()

        gen = MakeHumanMeshGenerator(makehuman_path=mh_dir)

        # Set up mock trimesh
        exported_files: list[str] = []

        class FakeTrimesh:
            def __init__(self, vertices: Any = None, faces: Any = None) -> None:
                """Initialize fake trimesh with vertices and faces."""
                self.vertices = vertices
                self.faces = faces

            def export(self, path: str) -> None:
                """Export."""
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).touch()
                exported_files.append(path)

            @property
            def convex_hull(self) -> FakeTrimesh:
                """Convex hull."""
                return self

        mock_trimesh.Trimesh = FakeTrimesh

        # Mock _run_makehuman_script to simulate success
        def mock_run(script_path: Path, timeout: int = 120) -> bool:
            """Simulate a successful MakeHuman script execution."""
            # Create fake OBJ output
            script_dir = script_path.parent
            obj_path = script_dir / "body.obj"
            groups_path = script_dir / "groups.json"

            # Write a simple OBJ
            obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            obj_path.write_text(obj_content, encoding="utf-8")

            # Write vertex groups
            groups = {"head": [0, 1, 2]}
            groups_path.write_text(json.dumps(groups), encoding="utf-8")
            return True

        with patch.object(gen, "_run_makehuman_script", side_effect=mock_run):
            result = gen.generate(_default_params(), tmp_path / "output")

        assert result.solver_status == "success"
        assert result.metadata["backend"] == "makehuman"

    def test_generate_fails_when_script_fails(self, tmp_path: Path) -> None:
        """Test that generation fails gracefully when MakeHuman script fails."""
        mh_dir = tmp_path / "makehuman"
        mh_dir.mkdir()
        (mh_dir / "makehuman.py").touch()

        gen = MakeHumanMeshGenerator(makehuman_path=mh_dir)

        with patch.object(gen, "_run_makehuman_script", return_value=False):
            result = gen.generate(_default_params(), tmp_path / "output")

        assert result.solver_status != "success"
        assert "failed" in result.error_message.lower()


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
