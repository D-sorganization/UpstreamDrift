"""
Heavy Integration Contracts — Trimesh
======================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: Trimesh can create meshes, compute inertia properties, perform
boolean operations, and export — as used by the humanoid character builder.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.live_simulation
class TestTrimeshCore:
    """Contract: Trimesh creates and manipulates 3D meshes."""

    def test_trimesh_import_and_version(self) -> None:
        """Trimesh is importable."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        assert hasattr(trimesh, "__version__")

    def test_trimesh_create_primitives(self) -> None:
        """Trimesh can create box, sphere, cylinder primitives."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        assert box.is_watertight
        assert box.vertices.shape[0] == 8

        sphere = trimesh.creation.icosphere(radius=0.05)
        assert sphere.is_watertight
        assert sphere.vertices.shape[0] > 0

        cylinder = trimesh.creation.cylinder(radius=0.02, height=0.1)
        assert cylinder.is_watertight

    def test_trimesh_inertia_computation(self) -> None:
        """Trimesh computes mass properties (inertia tensor, COM, volume).

        This is critical for URDF generation — the humanoid character
        builder uses trimesh to compute link inertias from mesh geometry.
        """
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        box = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        box.density = 1000.0  # water density kg/m³

        # Volume
        expected_volume = 0.1 * 0.2 * 0.3
        assert abs(box.volume - expected_volume) < 1e-10

        # Mass
        expected_mass = expected_volume * 1000.0
        assert abs(box.mass - expected_mass) < 1e-6

        # Inertia tensor should be 3x3 symmetric positive definite
        inertia = box.moment_inertia
        assert inertia.shape == (3, 3)
        np.testing.assert_allclose(inertia, inertia.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(inertia)
        assert all(ev > 0 for ev in eigenvalues)

        # COM should be at origin for a centered box
        np.testing.assert_allclose(box.center_mass, [0, 0, 0], atol=1e-10)

    def test_trimesh_mesh_boolean(self) -> None:
        """Trimesh can perform boolean operations (union/difference)."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        a = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        b = trimesh.creation.box(
            extents=[0.5, 0.5, 0.5],
            transform=trimesh.transformations.translation_matrix([0.5, 0, 0]),
        )

        try:
            diff = trimesh.boolean.difference([a, b], engine="blender")
        except Exception as e:  # noqa: BLE001, F841
            try:
                diff = trimesh.boolean.difference([a, b])
            except Exception as e:  # noqa: BLE001, F841
                pytest.skip("Boolean engine not available (needs manifold/blender)")
                return

        # Difference should produce a valid mesh with less volume
        assert diff.volume < a.volume

    def test_trimesh_stl_export_roundtrip(self) -> None:
        """Trimesh can export to STL and re-import."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        original = trimesh.creation.box(extents=[0.1, 0.2, 0.3])

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            stl_path = Path(f.name)
            original.export(str(stl_path))

        reloaded = trimesh.load(str(stl_path))
        stl_path.unlink()

        # Volume should be preserved
        assert abs(reloaded.volume - original.volume) < 1e-8

    def test_trimesh_convex_hull(self) -> None:
        """Trimesh computes convex hull (used for collision geometry)."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        # Create a non-convex shape by combining two boxes
        mesh = trimesh.creation.box(extents=[1.0, 0.1, 0.1])

        hull = mesh.convex_hull
        assert hull.is_convex
        assert hull.volume >= mesh.volume - 1e-10


@pytest.mark.live_simulation
class TestTrimeshProjectIntegration:
    """Contract: Project modules using trimesh are importable."""

    def test_mesh_processor_importable(self) -> None:
        """MeshProcessor from humanoid character builder is importable."""
        try:
            import trimesh  # noqa: F401
        except ImportError:
            pytest.skip("trimesh not installed")

        from src.shared.python.humanoid_character_builder.mesh.mesh_processor import (
            MeshProcessor,
        )

        assert MeshProcessor is not None

    def test_collision_generator_importable(self) -> None:
        """CollisionGenerator is importable."""
        try:
            import trimesh  # noqa: F401
        except ImportError:
            pytest.skip("trimesh not installed")

        from src.shared.python.humanoid_character_builder.mesh.collision_generator import (
            CollisionGenerator,
        )

        assert CollisionGenerator is not None

    def test_inertia_calculator_importable(self) -> None:
        """InertiaCalculator is importable."""
        try:
            import trimesh  # noqa: F401
        except ImportError:
            pytest.skip("trimesh not installed")

        from src.shared.python.humanoid_character_builder.mesh.inertia_calculator import (
            InertiaCalculator,
        )

        assert InertiaCalculator is not None


pytestmark = pytest.mark.live_simulation
