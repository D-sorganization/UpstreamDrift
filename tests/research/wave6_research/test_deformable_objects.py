"""Tests for src/research/deformable/objects.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.research.deformable.objects import (
    Cable,
    Cloth,
    MaterialProperties,
    SoftBody,
)

pytestmark = pytest.mark.unit


def _reference_soft_body_forces(soft_body: SoftBody) -> np.ndarray:
    forces = np.zeros_like(soft_body._mesh)
    mu = soft_body._material.shear_modulus
    lam = (
        soft_body._material.youngs_modulus
        * soft_body._material.poisson_ratio
        / (
            (1 + soft_body._material.poisson_ratio)
            * (1 - 2 * soft_body._material.poisson_ratio)
        )
    )

    for i, tet in enumerate(soft_body._tetrahedra):
        v0 = soft_body._mesh[tet[0]]
        v1 = soft_body._mesh[tet[1]]
        v2 = soft_body._mesh[tet[2]]
        v3 = soft_body._mesh[tet[3]]

        D = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
        F = D @ soft_body._B_matrices[i]
        J = np.linalg.det(F)
        if J <= 0:
            J = 0.01

        F_inv_t = np.linalg.inv(F).T
        P = mu * (F - F_inv_t) + lam * np.log(J) * F_inv_t
        H = -soft_body._rest_volumes[i] * P @ soft_body._B_matrices[i].T

        forces[tet[1]] += H[:, 0]
        forces[tet[2]] += H[:, 1]
        forces[tet[3]] += H[:, 2]
        forces[tet[0]] -= H[:, 0] + H[:, 1] + H[:, 2]

    return forces


def _reference_cable_forces(cable: Cable) -> np.ndarray:
    forces = np.zeros_like(cable._mesh)
    k_stretch = cable._material.youngs_modulus
    k_bend = cable._material.bending_stiffness or k_stretch * 0.1

    for i in range(len(cable._mesh) - 1):
        delta = cable._mesh[i + 1] - cable._mesh[i]
        length = math.hypot(delta[0], delta[1], delta[2])

        if length > 1e-10:
            direction = delta / length
            strain = (length - cable._rest_lengths[i]) / cable._rest_lengths[i]
            force_mag = k_stretch * strain
            force = force_mag * direction
            forces[i] += force
            forces[i + 1] -= force

    for i in range(1, len(cable._mesh) - 1):
        v1 = cable._mesh[i] - cable._mesh[i - 1]
        v2 = cable._mesh[i + 1] - cable._mesh[i]
        l1 = math.hypot(v1[0], v1[1], v1[2])
        l2 = math.hypot(v2[0], v2[1], v2[2])

        if l1 > 1e-10 and l2 > 1e-10:
            cos_angle = np.dot(v1, v2) / (l1 * l2)
            cos_angle = np.clip(cos_angle, -1, 1)
            bend_force = k_bend * (1 - cos_angle)
            direction = v2 / l2 - v1 / l1
            direction_norm = math.hypot(direction[0], direction[1], direction[2])

            if direction_norm > 1e-10:
                forces[i] -= bend_force * direction / direction_norm

    return forces


def _reference_cloth_forces(cloth: Cloth) -> np.ndarray:
    forces = np.zeros_like(cloth._mesh)
    k_stretch = cloth._material.youngs_modulus
    k_shear = cloth._material.shear_stiffness or k_stretch * 0.5
    k_bend = cloth._material.bending_stiffness or k_stretch * 0.1

    for i, j, rest_length, spring_type in cloth._springs:
        delta = cloth._mesh[j] - cloth._mesh[i]
        length = math.hypot(delta[0], delta[1], delta[2])

        if length < 1e-10:
            continue

        if spring_type == "stretch":
            k = k_stretch
        elif spring_type == "shear":
            k = k_shear
        else:
            k = k_bend

        direction = delta / length
        strain = length - rest_length
        force = k * strain * direction

        forces[i] += force
        forces[j] -= force

    return forces


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

    def test_compute_internal_forces_matches_scalar_reference_for_multiple_tets(
        self,
    ) -> None:
        mesh = np.array(
            [
                [0.00, 0.00, 0.00],
                [1.12, 0.05, 0.02],
                [0.04, 1.03, 0.06],
                [0.02, 0.08, 0.94],
                [1.06, 1.12, 0.88],
            ],
            dtype=float,
        )
        rest_mesh = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)
        sb = SoftBody(
            rest_mesh,
            tets,
            MaterialProperties(youngs_modulus=850.0, poisson_ratio=0.28),
        )
        sb.set_node_positions(mesh)

        np.testing.assert_allclose(
            sb.compute_internal_forces(),
            _reference_soft_body_forces(sb),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_compute_internal_forces_batches_deformation_inversion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mesh = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)
        sb = SoftBody(mesh, tets, MaterialProperties())
        original_inv = np.linalg.inv
        calls: list[tuple[int, ...]] = []

        def counting_inv(matrix: np.ndarray) -> np.ndarray:
            calls.append(matrix.shape)
            return original_inv(matrix)

        monkeypatch.setattr(np.linalg, "inv", counting_inv)
        sb.compute_internal_forces()

        assert calls == [(2, 3, 3)]

    def test_compute_internal_forces_batches_root_force_reduction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mesh = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int)
        sb = SoftBody(mesh, tets, MaterialProperties())
        original_einsum = np.einsum
        calls: list[tuple[str, tuple[int, ...]]] = []

        def counting_einsum(subscripts: str, operand: np.ndarray) -> np.ndarray:
            calls.append((subscripts, operand.shape))
            return original_einsum(subscripts, operand)

        monkeypatch.setattr(np, "einsum", counting_einsum)
        sb.compute_internal_forces()

        assert calls == [("ijk->ij", (2, 3, 3))]


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

    def test_compute_internal_forces_matches_scalar_reference(self) -> None:
        mesh = np.array(
            [
                [0.00, 0.00, 0.00],
                [1.08, 0.06, 0.02],
                [2.03, 0.21, 0.10],
                [2.88, 0.29, 0.31],
                [4.02, 0.45, 0.27],
            ],
            dtype=float,
        )
        rest = np.array([1.0, 0.95, 0.9, 1.05])
        c = Cable(
            mesh,
            MaterialProperties(youngs_modulus=120.0, bending_stiffness=7.5),
            rest_lengths=rest,
        )

        np.testing.assert_allclose(
            c.compute_internal_forces(),
            _reference_cable_forces(c),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_spring_connectivity_is_cached_as_vectors(self) -> None:
        mesh = np.array([[0, 0, 0], [1.1, 0, 0], [2.1, 0.1, 0]], dtype=float)
        c = Cable(mesh, MaterialProperties())

        assert c._spring_i.tolist() == [0, 1]
        assert c._spring_j.tolist() == [1, 2]
        np.testing.assert_allclose(c._spring_rest_lengths, c._rest_lengths)


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

    def test_compute_internal_forces_matches_scalar_reference(self) -> None:
        mesh = self._grid_mesh(4, 3)
        mesh[:, 2] = np.array(
            [0.00, 0.04, -0.02, 0.03, 0.02, 0.08, 0.03, -0.01, 0.01, 0.05, 0.09, 0.04]
        )
        mesh[5, :2] += np.array([0.07, -0.03])
        mesh[10, :2] += np.array([-0.05, 0.04])
        cl = Cloth(
            mesh,
            4,
            3,
            MaterialProperties(
                youngs_modulus=140.0,
                shear_stiffness=44.0,
                bending_stiffness=8.0,
            ),
        )
        deformed = mesh.copy()
        deformed[2] += np.array([0.04, -0.02, 0.05])
        deformed[5] += np.array([-0.03, 0.06, -0.01])
        deformed[10] += np.array([0.02, 0.03, 0.04])
        cl.set_node_positions(deformed)

        np.testing.assert_allclose(
            cl.compute_internal_forces(),
            _reference_cloth_forces(cl),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_spring_connectivity_is_cached_as_vectors(self) -> None:
        mesh = self._grid_mesh(3, 3)
        cl = Cloth(mesh, 3, 3, MaterialProperties())

        assert cl._spring_i.shape == cl._spring_j.shape == cl._spring_rest_lengths.shape
        assert cl._spring_i.shape == cl._spring_stiffness.shape
        assert cl._spring_i.size == len(cl._springs)
        np.testing.assert_array_equal(
            cl._spring_i,
            np.array([i for i, _, _, _ in cl._springs], dtype=np.intp),
        )
