from typing import Any

"""test_collision_geometry.py module."""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
from humanoid_character_builder.mesh.collision_geometry import (
    CollisionGeometryGenerator,
)


@pytest.fixture
def generator() -> Any:
    return CollisionGeometryGenerator()


@pytest.fixture
def box_mesh() -> Any:
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


@pytest.fixture
def sphere_mesh() -> Any:
    return trimesh.creation.icosphere(radius=1.0, subdivisions=2)


def test_generate_primitives_box(generator, box_mesh) -> Any:
    result = generator.generate(box_mesh, method="primitives")
    assert len(result.meshes) == 1
    # Check if volume is close
    # Box volume = 1.0
    assert np.isclose(result.meshes[0].volume, 1.0, rtol=0.1)
    assert result.method == "primitives"


def test_generate_vhacd_fallback(generator, sphere_mesh) -> Any:
    # VHACD likely falls back to hull if binary missing
    result = generator.generate(sphere_mesh, method="vhacd")
    assert len(result.meshes) >= 1
    assert result.method == "vhacd"
    # Even if fallback, it should produce a valid mesh
    assert result.meshes[0].volume > 0


def test_generate_decimation(generator, sphere_mesh) -> Any:
    # Sphere has 80 faces at subdiv 2
    target = 20
    result = generator.generate(sphere_mesh, method="decimation", max_triangles=target)
    assert len(result.meshes) == 1
    # Decimation is approximate
    assert len(result.meshes[0].faces) < len(sphere_mesh.faces)
    assert result.method == "decimation"


def test_generate_auto(generator, sphere_mesh) -> Any:
    result = generator.generate(sphere_mesh, method="auto")
    assert len(result.meshes) >= 1
    assert result.method == "auto"


def test_quality_metrics(generator, box_mesh) -> Any:
    result = generator.generate(box_mesh, method="primitives")
    # Should be nearly identical
    assert result.quality_score > 0.8
    assert result.volume_preservation > 0.9
    assert result.vertex_count > 0
    assert result.face_count > 0
    assert result.processing_time >= 0


def test_generate_cylinder_fit(generator) -> Any:
    # Create a cylinder and see if it fits a cylinder
    cyl = trimesh.creation.cylinder(radius=0.5, height=2.0)
    result = generator.generate(cyl, method="primitives")
    # Volume of cyl = pi * 0.5^2 * 2 = 1.57
    assert np.isclose(result.meshes[0].volume, cyl.volume, rtol=0.2)


def test_generate_capsule_fit(generator) -> Any:
    # Create a capsule
    cap = trimesh.creation.capsule(radius=0.5, height=2.0)
    result = generator.generate(cap, method="primitives")
    # Should pick capsule or very close primitive
    assert len(result.meshes) == 1
    assert np.isclose(result.meshes[0].volume, cap.volume, rtol=0.2)


def test_generate_oriented_cylinder(generator) -> Any:
    # Cylinder along X axis
    cyl = trimesh.creation.cylinder(radius=0.5, height=2.0)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    result = generator.generate(cyl, method="primitives")
    assert len(result.meshes) == 1
    assert np.isclose(result.meshes[0].volume, cyl.volume, rtol=0.2)
