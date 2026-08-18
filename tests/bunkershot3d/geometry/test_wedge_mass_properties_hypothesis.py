"""Metamorphic property tests for the mass-property integrator (#8609).

A rigid motion of the mesh must leave the volume invariant, carry the
centroid with it, and conjugate the inertia tensor: I -> R I R^T.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from bunkershot3d.geometry.mass_properties import compute_mass_properties
from bunkershot3d.geometry.solids import box_mesh, cylinder_mesh, tetrahedron_mesh

pytestmark = pytest.mark.unit

SOLVER_SETTINGS = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

finite_floats = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _meshes() -> list:  # type: ignore[type-arg]
    return [
        box_mesh(0.03, 0.02, 0.01),
        cylinder_mesh(0.012, 0.03, n_segments=24),
        tetrahedron_mesh(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.021, 0.003, 0.0],
                    [0.004, 0.030, 0.002],
                    [0.001, 0.002, 0.040],
                ]
            )
        ),
    ]


@pytest.mark.parametrize("mesh_index", [0, 1, 2])
@given(quaternion=st.lists(finite_floats, min_size=4, max_size=4))
@SOLVER_SETTINGS
def test_rigid_rotation_conjugates_the_inertia_tensor(
    mesh_index: int, quaternion: list[float]
) -> None:
    quat = np.asarray(quaternion, dtype=np.float64)
    assume(np.linalg.norm(quat) > 0.2)
    rotation = _quaternion_to_rotation(quat)

    mesh = _meshes()[mesh_index]
    base = compute_mass_properties(mesh, mass_kg=0.3)
    turned = compute_mass_properties(mesh.transformed(rotation=rotation), mass_kg=0.3)

    assert turned.volume_m3 == pytest.approx(base.volume_m3, rel=1e-11)
    np.testing.assert_allclose(
        turned.centroid_m, rotation @ base.centroid_m, rtol=1e-9, atol=1e-15
    )
    np.testing.assert_allclose(
        turned.inertia_kg_m2,
        rotation @ base.inertia_kg_m2 @ rotation.T,
        rtol=1e-9,
        atol=1e-17,
    )


@pytest.mark.parametrize("mesh_index", [0, 1, 2])
@given(
    quaternion=st.lists(finite_floats, min_size=4, max_size=4),
    translation=st.lists(finite_floats, min_size=3, max_size=3),
)
@SOLVER_SETTINGS
def test_full_rigid_motion(
    mesh_index: int, quaternion: list[float], translation: list[float]
) -> None:
    quat = np.asarray(quaternion, dtype=np.float64)
    assume(np.linalg.norm(quat) > 0.2)
    rotation = _quaternion_to_rotation(quat)
    shift = np.asarray(translation, dtype=np.float64) * 0.1

    mesh = _meshes()[mesh_index]
    base = compute_mass_properties(mesh, mass_kg=0.3)
    moved = compute_mass_properties(
        mesh.transformed(rotation=rotation, translation=shift), mass_kg=0.3
    )

    assert moved.volume_m3 == pytest.approx(base.volume_m3, rel=1e-10)
    np.testing.assert_allclose(
        moved.centroid_m, rotation @ base.centroid_m + shift, rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(
        moved.inertia_kg_m2,
        rotation @ base.inertia_kg_m2 @ rotation.T,
        rtol=1e-7,
        atol=1e-15,
    )


@pytest.mark.parametrize("mesh_index", [0, 1, 2])
@given(scale=st.floats(min_value=0.2, max_value=5.0, allow_subnormal=False))
@SOLVER_SETTINGS
def test_uniform_scaling_follows_the_dimensional_law(
    mesh_index: int, scale: float
) -> None:
    """Volume scales as lam^3 and inertia (at fixed mass) as lam^2."""
    mesh = _meshes()[mesh_index]
    base = compute_mass_properties(mesh, mass_kg=0.3)
    scaled = compute_mass_properties(
        mesh.transformed(rotation=scale * np.eye(3), check_orthogonal=False),
        mass_kg=0.3,
    )
    assert scaled.volume_m3 == pytest.approx(base.volume_m3 * scale**3, rel=1e-10)
    np.testing.assert_allclose(
        scaled.inertia_kg_m2, base.inertia_kg_m2 * scale**2, rtol=1e-9, atol=1e-17
    )
