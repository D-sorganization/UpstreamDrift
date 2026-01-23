"""Tests for ellipsoid visualization module.

Guideline I (Mobility and Force Ellipsoids) implementation tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.shared.python.ellipsoid_visualization import (
    EllipsoidData,
    EllipsoidSequence,
    EllipsoidVisualizer,
    compute_force_ellipsoid,
    compute_velocity_ellipsoid,
    ellipsoid_to_json,
    export_ellipsoid_obj,
    export_ellipsoid_sequence_json,
    export_ellipsoid_stl,
    generate_ellipsoid_mesh,
)

if TYPE_CHECKING:
    from shared.python.interfaces import PhysicsEngine


class TestEllipsoidComputation:
    """Tests for ellipsoid computation functions."""

    @pytest.fixture
    def mock_engine(self) -> PhysicsEngine:
        """Create a mock engine that returns a known Jacobian."""

        class MockEngine:
            def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
                if body_name == "test_body":
                    # 3x2 Jacobian for testing
                    J = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
                    return {
                        "linear": J,
                        "angular": np.zeros((3, 2)),
                        "spatial": np.vstack([np.zeros((3, 2)), J]),
                    }
                return None

            def get_time(self) -> float:
                return 0.5

        return MockEngine()  # type: ignore[return-value]

    def test_velocity_ellipsoid_computation(self, mock_engine: PhysicsEngine) -> None:
        """Velocity ellipsoid should have correct radii from SVD."""
        ellipsoid = compute_velocity_ellipsoid(mock_engine, "test_body")

        assert ellipsoid is not None
        assert ellipsoid.ellipsoid_type == "velocity"
        assert ellipsoid.body_name == "test_body"
        assert ellipsoid.timestep == 0.5

        # Radii should be singular values of J
        # J = [[1, 0], [0, 2], [0, 0]]
        # SVD gives singular values [2.0, 1.0]
        np.testing.assert_allclose(
            sorted(ellipsoid.radii, reverse=True), [2.0, 1.0], atol=1e-10
        )

    def test_force_ellipsoid_has_inverse_radii(
        self, mock_engine: PhysicsEngine
    ) -> None:
        """Force ellipsoid radii should be inverse of velocity radii."""
        vel_ellipsoid = compute_velocity_ellipsoid(mock_engine, "test_body")
        force_ellipsoid = compute_force_ellipsoid(mock_engine, "test_body")

        assert vel_ellipsoid is not None
        assert force_ellipsoid is not None

        # Force radii = 1 / velocity radii
        for vr, fr in zip(vel_ellipsoid.radii, force_ellipsoid.radii, strict=True):
            if vr > 1e-15:
                np.testing.assert_allclose(fr, 1.0 / vr, atol=1e-10)

    def test_unknown_body_returns_none(self, mock_engine: PhysicsEngine) -> None:
        """Unknown body name should return None."""
        ellipsoid = compute_velocity_ellipsoid(mock_engine, "nonexistent_body")
        assert ellipsoid is None

    def test_condition_number_computed(self, mock_engine: PhysicsEngine) -> None:
        """Condition number should be computed correctly."""
        ellipsoid = compute_velocity_ellipsoid(mock_engine, "test_body")

        assert ellipsoid is not None
        # κ = σ_max / σ_min = 2.0 / 1.0 = 2.0
        np.testing.assert_allclose(ellipsoid.condition_number, 2.0, atol=1e-10)


class TestEllipsoidMesh:
    """Tests for ellipsoid mesh generation."""

    def test_mesh_generation_produces_valid_mesh(self) -> None:
        """Generated mesh should have valid vertices and faces."""
        ellipsoid = EllipsoidData(
            center=np.array([0.0, 0.0, 0.0]),
            radii=np.array([1.0, 2.0, 0.5]),
            axes=np.eye(3),
            body_name="test",
        )

        vertices, faces = generate_ellipsoid_mesh(ellipsoid)

        # Should have vertices
        assert vertices.shape[0] > 0
        assert vertices.shape[1] == 3

        # Should have faces
        assert faces.shape[0] > 0
        assert faces.shape[1] == 3

        # All face indices should be valid
        assert np.all(faces >= 0)
        assert np.all(faces < vertices.shape[0])

    def test_mesh_has_correct_extent(self) -> None:
        """Mesh should have extent matching ellipsoid radii."""
        radii = np.array([2.0, 3.0, 1.0])
        ellipsoid = EllipsoidData(
            center=np.array([0.0, 0.0, 0.0]),
            radii=radii,
            axes=np.eye(3),
            body_name="test",
        )

        vertices, _ = generate_ellipsoid_mesh(ellipsoid)

        # Max extent in each axis should approximately match radii
        max_extent = np.max(np.abs(vertices), axis=0)
        np.testing.assert_allclose(max_extent, radii, atol=0.1)


class TestEllipsoidExport:
    """Tests for ellipsoid export functions."""

    def test_json_serialization(self) -> None:
        """EllipsoidData should serialize to JSON correctly."""
        ellipsoid = EllipsoidData(
            center=np.array([1.0, 2.0, 3.0]),
            radii=np.array([0.5, 1.0, 1.5]),
            axes=np.eye(3),
            body_name="test",
            ellipsoid_type="velocity",
            condition_number=3.0,
            timestep=0.1,
        )

        json_data = ellipsoid_to_json(ellipsoid)

        assert json_data["body_name"] == "test"
        assert json_data["ellipsoid_type"] == "velocity"
        assert json_data["condition_number"] == 3.0
        assert json_data["timestep"] == 0.1
        assert json_data["center"] == [1.0, 2.0, 3.0]

    def test_sequence_json_export(self, tmp_path: Path) -> None:
        """EllipsoidSequence should export to JSON file."""
        ellipsoids = [
            EllipsoidData(
                center=np.array([0.0, 0.0, 0.0]),
                radii=np.array([1.0, 1.0, 1.0]),
                axes=np.eye(3),
                body_name="test",
                timestep=float(i) * 0.1,
            )
            for i in range(3)
        ]

        sequence = EllipsoidSequence(
            ellipsoids=ellipsoids,
            timesteps=np.array([0.0, 0.1, 0.2]),
            body_name="test",
        )

        output_file = tmp_path / "test_sequence.json"
        export_ellipsoid_sequence_json(sequence, output_file)

        assert output_file.exists()

        # Verify file can be parsed
        import json

        with open(output_file) as f:
            data = json.load(f)

        assert len(data["ellipsoids"]) == 3
        assert data["body_name"] == "test"

    def test_obj_export(self, tmp_path: Path) -> None:
        """Ellipsoid should export to valid OBJ file."""
        ellipsoid = EllipsoidData(
            center=np.array([0.0, 0.0, 0.0]),
            radii=np.array([1.0, 2.0, 0.5]),
            axes=np.eye(3),
            body_name="test",
        )

        output_file = tmp_path / "test_ellipsoid.obj"
        export_ellipsoid_obj(ellipsoid, output_file)

        assert output_file.exists()

        # Verify file has vertices and faces
        content = output_file.read_text()
        assert "v " in content  # Has vertices
        assert "f " in content  # Has faces

    def test_stl_binary_export(self, tmp_path: Path) -> None:
        """Ellipsoid should export to valid binary STL file."""
        ellipsoid = EllipsoidData(
            center=np.array([0.0, 0.0, 0.0]),
            radii=np.array([1.0, 2.0, 0.5]),
            axes=np.eye(3),
            body_name="test",
            ellipsoid_type="velocity",
        )

        output_file = tmp_path / "test_ellipsoid.stl"
        export_ellipsoid_stl(ellipsoid, output_file, binary=True)

        assert output_file.exists()
        # Binary STL has 80-byte header + 4 bytes for triangle count
        assert output_file.stat().st_size >= 84

    def test_stl_ascii_export(self, tmp_path: Path) -> None:
        """Ellipsoid should export to valid ASCII STL file."""
        ellipsoid = EllipsoidData(
            center=np.array([0.0, 0.0, 0.0]),
            radii=np.array([1.0, 2.0, 0.5]),
            axes=np.eye(3),
            body_name="test",
            ellipsoid_type="force",
        )

        output_file = tmp_path / "test_ellipsoid_ascii.stl"
        export_ellipsoid_stl(ellipsoid, output_file, binary=False)

        assert output_file.exists()

        # Verify ASCII STL structure
        content = output_file.read_text()
        assert "solid " in content
        assert "facet normal" in content
        assert "vertex" in content
        assert "endsolid" in content


class TestEllipsoidVisualizer:
    """Tests for EllipsoidVisualizer class."""

    @pytest.fixture
    def mock_engine(self) -> PhysicsEngine:
        """Create a mock engine for visualizer testing."""

        class MockEngine:
            def __init__(self) -> None:
                self._time = 0.0

            def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
                J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
                return {"linear": J, "angular": np.zeros((3, 2)), "spatial": J}

            def get_time(self) -> float:
                return self._time

            def set_time(self, t: float) -> None:
                self._time = t

        return MockEngine()  # type: ignore[return-value]

    def test_update_ellipsoids(self, mock_engine: PhysicsEngine) -> None:
        """Visualizer should update ellipsoids for requested bodies."""
        viz = EllipsoidVisualizer(mock_engine)

        results = viz.update_ellipsoids(["test_body"])

        assert "test_body" in results
        assert "test_body" in viz.ellipsoid_cache

    def test_record_frame_creates_sequence(self, mock_engine: PhysicsEngine) -> None:
        """Recording frames should create EllipsoidSequence."""
        viz = EllipsoidVisualizer(mock_engine)

        mock_engine.set_time(0.0)  # type: ignore[attr-defined]
        viz.record_frame(["test_body"])

        mock_engine.set_time(0.1)  # type: ignore[attr-defined]
        viz.record_frame(["test_body"])

        assert "test_body" in viz.sequences
        assert len(viz.sequences["test_body"].ellipsoids) == 2
        viz.finalize_sequences()  # Must finalize to convert timestep list to array
        np.testing.assert_allclose(viz.sequences["test_body"].timesteps, [0.0, 0.1])

    def test_manipulability_summary(self, mock_engine: PhysicsEngine) -> None:
        """Summary should contain expected metrics."""
        viz = EllipsoidVisualizer(mock_engine)
        viz.update_ellipsoids(["test_body"])

        summary = viz.get_manipulability_summary("test_body")

        assert summary is not None
        assert "max_radius" in summary
        assert "min_radius" in summary
        assert "manipulability_index" in summary
        assert "condition_number" in summary
        assert "isotropy" in summary
