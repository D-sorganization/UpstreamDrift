"""Tests for src/research/deformable/objects.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.research.deformable.objects import (
    Cable,
    Cloth,
    MaterialProperties,
    SoftBody,
)


class TestGuards:
    def test_soft_body_none_mesh(self) -> None:
        with pytest.raises(ValueError, match="mesh"):
            SoftBody(None, np.array([[0, 1, 2, 3]]), MaterialProperties())  # type: ignore[arg-type]

    def test_cable_none_mesh(self) -> None:
        with pytest.raises(ValueError, match="mesh"):
            Cable(None, MaterialProperties())  # type: ignore[arg-type]

    def test_cloth_none_mesh(self) -> None:
        with pytest.raises(ValueError, match="mesh"):
            Cloth(None, 2, 2, MaterialProperties())  # type: ignore[arg-type]


class TestMaterialProperties:
    def test_defaults(self) -> None:
        mp = MaterialProperties()
        assert mp.youngs_modulus == 1e6
        assert mp.poisson_ratio == 0.3
        assert mp.density == 1000.0
        assert mp.bending_stiffness is None

    def test_shear_modulus(self) -> None:
        mp = MaterialProperties(youngs_modulus=2.6e6, poisson_ratio=0.3)
        assert mp.shear_modulus == pytest.approx(2.6e6 / (2 * 1.3))

    def test_bulk_modulus(self) -> None:
        mp = MaterialProperties(youngs_modulus=3e6, poisson_ratio=0.2)
        assert mp.bulk_modulus == pytest.approx(3e6 / (3 * 0.6))


@pytest.fixture
def tet_mesh() -> tuple[np.ndarray, np.ndarray]:
    # Single tetrahedron
    mesh = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tets = np.array([[0, 1, 2, 3]])
    return mesh, tets


class TestSoftBody:
    def test_construction(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        assert sb.n_nodes == 4
        np.testing.assert_allclose(sb.mesh, mesh)
        assert sb.material.density == 1000.0
        # rest volume = 1/6 for unit tet
        assert sb._rest_volumes[0] == pytest.approx(1 / 6)

    def test_get_set_positions(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        pos = sb.get_node_positions()
        assert pos.shape == (4, 3)
        new_pos = pos + 0.5
        sb.set_node_positions(new_pos)
        np.testing.assert_allclose(sb.get_node_positions(), new_pos)

    def test_velocities_start_zero(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        np.testing.assert_allclose(sb.get_node_velocities(), 0.0)

    def test_apply_external_force_array(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        sb.apply_external_force([0, 1], np.array([[1.0, 0, 0], [0, 1, 0]]))
        assert sb._external_forces[0, 0] == 1.0
        assert sb._external_forces[1, 1] == 1.0

    def test_apply_external_force_broadcast(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        sb.apply_external_force([0, 1, 2], np.array([0.0, 0, -1.0]))
        assert sb._external_forces[0, 2] == -1.0
        assert sb._external_forces[2, 2] == -1.0

    def test_clear_external_forces(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        sb.apply_external_force([0], np.array([1.0, 1, 1]))
        sb.clear_external_forces()
        np.testing.assert_allclose(sb._external_forces, 0.0)

    def test_fix_unfix_nodes(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        sb.fix_nodes([0, 1])
        assert sb._fixed_nodes == {0, 1}
        sb.unfix_nodes([0])
        assert sb._fixed_nodes == {1}

    def test_compute_internal_forces_at_rest(self, tet_mesh) -> None:
        # At rest, F=I -> P=0 -> forces approximately zero
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        forces = sb.compute_internal_forces()
        assert forces.shape == (4, 3)
        # neo-hookean at F=I gives mu*(I - I^-T) + lam*log(1)*I^-T = 0
        np.testing.assert_allclose(forces, 0.0, atol=1e-8)

    def test_step_stable(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties(damping=0.5))
        sb.fix_nodes([0])
        sb.step(1e-4)
        # node 0 stays fixed
        np.testing.assert_allclose(sb.mesh[0], mesh[0])

    def test_reset(self, tet_mesh) -> None:
        mesh, tets = tet_mesh
        sb = SoftBody(mesh, tets, MaterialProperties())
        sb.set_node_positions(mesh + 5)
        sb._velocities += 2.0
        sb.reset()
        np.testing.assert_allclose(sb.mesh, mesh)
        np.testing.assert_allclose(sb.get_node_velocities(), 0.0)

    def test_degenerate_shape_matrix(self) -> None:
        # Degenerate tet (coplanar) should not crash
        mesh = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        tets = np.array([[0, 1, 2, 3]])
        sb = SoftBody(mesh, tets, MaterialProperties())
        assert sb._B_matrices[0].shape == (3, 3)


class TestCable:
    def test_construction(self) -> None:
        mesh = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        c = Cable(mesh, MaterialProperties())
        assert c.n_nodes == 3
        assert c.rest_length == pytest.approx(2.0)

    def test_custom_rest_lengths(self) -> None:
        mesh = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        rest = np.array([0.5, 0.5])
        c = Cable(mesh, MaterialProperties(), rest_lengths=rest)
        assert c.rest_length == pytest.approx(1.0)

    def test_get_length_at_rest(self) -> None:
        mesh = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        c = Cable(mesh, MaterialProperties())
        assert c.get_length() == pytest.approx(2.0)

    def test_get_tension(self) -> None:
        mesh = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        c = Cable(mesh, MaterialProperties())
        # No stretch -> tension ~0
        assert c.get_tension() == pytest.approx(0.0)

    def test_compute_internal_forces_when_stretched(self) -> None:
        mesh = np.array([[0, 0, 0], [2.0, 0, 0]], dtype=float)
        rest = np.array([1.0])
        c = Cable(mesh, MaterialProperties(youngs_modulus=100.0), rest_lengths=rest)
        f = c.compute_internal_forces()
        # Pulls inward
        assert f[0, 0] > 0
        assert f[1, 0] < 0

    def test_step_with_fixed(self) -> None:
        mesh = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        c = Cable(mesh, MaterialProperties(damping=0.1))
        c.fix_nodes([0])
        c.step(1e-3)
        np.testing.assert_allclose(c.mesh[0], mesh[0])

    def test_bending_forces_in_straight_cable(self) -> None:
        mesh = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        c = Cable(mesh, MaterialProperties())
        f = c.compute_internal_forces()
        # Straight cable: cos_angle == 1, so bending contribution is zero
        assert np.allclose(f, 0.0)


class TestCloth:
    def _grid_mesh(self, w: int = 3, h: int = 3) -> np.ndarray:
        pts = []
        for y in range(h):
            for x in range(w):
                pts.append([float(x), float(y), 0.0])
        return np.array(pts)

    def test_construction(self) -> None:
        mesh = self._grid_mesh(3, 3)
        cl = Cloth(mesh, 3, 3, MaterialProperties())
        assert cl.width == 3
        assert cl.height == 3
        assert cl.n_nodes == 9
        # springs: stretch (12) + shear (8) + bend (6) = 26
        types = {t for _, _, _, t in cl._springs}
        assert types == {"stretch", "shear", "bend"}

    def test_compute_internal_forces_at_rest(self) -> None:
        mesh = self._grid_mesh(3, 3)
        cl = Cloth(mesh, 3, 3, MaterialProperties())
        f = cl.compute_internal_forces()
        np.testing.assert_allclose(f, 0.0, atol=1e-10)

    def test_step(self) -> None:
        mesh = self._grid_mesh(3, 3)
        cl = Cloth(mesh, 3, 3, MaterialProperties(damping=0.05))
        cl.fix_nodes([0])
        cl.step(1e-4)
        np.testing.assert_allclose(cl.mesh[0], mesh[0])

    def test_attach_to_body(self) -> None:
        mesh = self._grid_mesh(2, 2)
        cl = Cloth(mesh, 2, 2, MaterialProperties())
        new_pos = np.array([[5, 5, 5], [6, 5, 5]], dtype=float)
        cl.attach_to_body("b1", [0, 1], new_pos)
        np.testing.assert_allclose(cl.mesh[0], [5, 5, 5])
        assert 0 in cl._fixed_nodes
        assert 1 in cl._fixed_nodes

    def test_degenerate_spring_skipped(self) -> None:
        # Coincident nodes -> length<1e-10 -> skipped
        mesh = np.zeros((4, 3))
        cl = Cloth(mesh, 2, 2, MaterialProperties())
        f = cl.compute_internal_forces()
        np.testing.assert_allclose(f, 0.0)
