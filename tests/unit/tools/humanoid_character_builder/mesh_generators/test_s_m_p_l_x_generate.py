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


class TestSMPLXGenerate:
    """Test the full SMPL-X generate pipeline with mocked smplx module."""

    def _mock_smplx_output(
        self, n_verts: int = 10475, n_faces: int = 20000
    ) -> MagicMock:
        """Create a mock SMPL-X model output.

        Generates face indices that are coherent with SMPLX_SEGMENT_VERTEX_RANGES
        so that _segment_mesh can extract valid segments.
        """
        rng = np.random.default_rng(42)

        mock_output = MagicMock()
        mock_output.vertices = MagicMock()
        mock_output.vertices.detach.return_value.cpu.return_value.numpy.return_value.squeeze.return_value = rng.standard_normal(
            (n_verts, 3)
        ).astype(
            np.float32
        )

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        # Build faces that stay within segment vertex ranges so _segment_mesh
        # can find them.  We allocate ~n_faces total across all segments.
        ranges = list(SMPLXMeshGenerator.SMPLX_SEGMENT_VERTEX_RANGES.values())
        faces_per_seg = max(1, n_faces // len(ranges))
        all_faces: list[np.ndarray] = []
        for start, end in ranges:
            seg_size = end - start
            if seg_size < 3:
                continue
            seg_faces = rng.integers(start, end, size=(faces_per_seg, 3))
            all_faces.append(seg_faces)
        mock_model.faces = np.vstack(all_faces).astype(np.int64)

        # Make lbs_weights raise AttributeError so fallback path is used
        del mock_model.lbs_weights

        return mock_model

    @patch("humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE", True)
    @patch(
        "humanoid_character_builder.generators.mesh_generator.TRIMESH_AVAILABLE", True
    )
    @patch("humanoid_character_builder.generators.mesh_generator._smplx_module")
    @patch("humanoid_character_builder.generators.mesh_generator._trimesh_module")
    def test_generate_produces_stl_files(
        self, mock_trimesh, mock_smplx, tmp_path: Path
    ) -> None:
        """Verify that generate produces per-segment STL files.

        Does NOT require torch — the production code in _mesh_smplx.py has
        a numpy fallback when torch is unavailable, and this test exercises
        that path. See issue #4543.
        """
        mock_model = self._mock_smplx_output()
        mock_smplx.create.return_value = mock_model

        # Mock trimesh.Trimesh to track exports
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

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        gen = SMPLXMeshGenerator(model_dir=model_dir)
        params = _default_params()
        output_dir = tmp_path / "output"

        result = gen.generate(params, output_dir)

        assert result.solver_status == "success"
        assert result.metadata["backend"] == "smplx"
        assert len(result.mesh_paths) > 0
        assert len(result.collision_paths) > 0
        assert len(result.vertex_groups) > 0

    @patch("humanoid_character_builder.generators.mesh_generator.SMPLX_AVAILABLE", True)
    @patch(
        "humanoid_character_builder.generators.mesh_generator.TRIMESH_AVAILABLE", True
    )
    def test_generate_handles_exception_gracefully(self, tmp_path: Path) -> None:
        """Verify graceful failure when SMPL-X forward pass throws."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        gen = SMPLXMeshGenerator(model_dir=model_dir)

        with patch(
            "humanoid_character_builder.generators.mesh_generator._smplx_module"
        ) as mock_smplx:
            mock_smplx.create.side_effect = RuntimeError("Model load failed")
            result = gen.generate(_default_params(), tmp_path / "out")

        assert result.solver_status != "success"
        assert "error" in result.error_message.lower()


# ---------------------------------------------------------------------------
# MakeHuman Generator Tests  (See issue #979)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GeneratedMeshResult Tests
# ---------------------------------------------------------------------------
