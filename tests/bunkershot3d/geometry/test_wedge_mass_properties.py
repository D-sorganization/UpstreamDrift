"""Native mass-property tests (issue #8609).

Volume, centroid and the full inertia tensor come from exact
divergence-theorem integration over the triangle mesh (numpy only).
Polyhedral solids (box, tetrahedron) are exact to floating-point
tolerance; curved solids (sphere, cylinder) carry a tessellation error
that must *converge*, which is tested explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.geometry.mass_properties import compute_mass_properties
from bunkershot3d.geometry.solids import (
    box_inertia_about_centroid,
    box_mesh,
    cylinder_inertia_about_centroid,
    cylinder_mesh,
    icosphere_mesh,
    sphere_inertia_about_centroid,
    tetrahedron_mesh,
)

pytestmark = pytest.mark.unit

FP_TOL = 1e-12


def _rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle_rad) * cross
        + (1.0 - np.cos(angle_rad)) * (cross @ cross)
    )


class TestApi:
    def test_requires_exactly_one_of_mass_or_density(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            compute_mass_properties(mesh)
        with pytest.raises(ValueError):
            compute_mass_properties(mesh, mass_kg=1.0, density_kg_m3=1.0)

    def test_mass_and_density_are_consistent(self) -> None:
        mesh = box_mesh(0.02, 0.03, 0.04)
        props = compute_mass_properties(mesh, mass_kg=0.304)
        assert props.mass_kg == pytest.approx(0.304)
        assert props.density_kg_m3 == pytest.approx(0.304 / props.volume_m3)

        by_density = compute_mass_properties(mesh, density_kg_m3=props.density_kg_m3)
        assert by_density.mass_kg == pytest.approx(0.304, rel=1e-12)

    def test_rejects_a_mesh_that_is_not_a_closed_solid(self) -> None:
        from bunkershot3d.geometry.mesh import MeshValidationError, TriangleMesh

        mesh = box_mesh(1.0, 1.0, 1.0)
        holed = TriangleMesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)[1:])
        with pytest.raises(MeshValidationError):
            compute_mass_properties(holed, density_kg_m3=1.0)

    def test_rejects_non_positive_density(self) -> None:
        with pytest.raises(ValueError):
            compute_mass_properties(box_mesh(1.0, 1.0, 1.0), density_kg_m3=0.0)


class TestBoxIsExact:
    def test_volume_centroid_and_inertia(self) -> None:
        lx, ly, lz, mass = 0.03, 0.02, 0.01, 0.3
        props = compute_mass_properties(box_mesh(lx, ly, lz), mass_kg=mass)
        assert props.volume_m3 == pytest.approx(lx * ly * lz, rel=FP_TOL)
        np.testing.assert_allclose(props.centroid_m, np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(
            props.inertia_kg_m2,
            box_inertia_about_centroid(mass, lx, ly, lz),
            rtol=FP_TOL,
            atol=1e-18,
        )

    def test_offset_box_uses_the_parallel_axis_theorem(self) -> None:
        lx, ly, lz, mass = 0.03, 0.02, 0.01, 0.3
        centre = np.array([0.11, -0.07, 0.05])
        props = compute_mass_properties(
            box_mesh(lx, ly, lz, centre=centre), mass_kg=mass
        )
        np.testing.assert_allclose(props.centroid_m, centre, atol=1e-15)
        np.testing.assert_allclose(
            props.inertia_kg_m2,
            box_inertia_about_centroid(mass, lx, ly, lz),
            rtol=1e-11,
            atol=1e-18,
        )
        expected_origin = props.inertia_kg_m2 + mass * (
            float(centre @ centre) * np.eye(3) - np.outer(centre, centre)
        )
        np.testing.assert_allclose(
            props.inertia_about_origin_kg_m2, expected_origin, rtol=1e-11, atol=1e-18
        )


class TestTetrahedronIsExact:
    """Closed form for the corner tetrahedron (0,0,0),(a,0,0),(0,b,0),(0,0,c).

    Over the standard simplex the normalised moments are <u_i^2> = 1/10 and
    <u_i u_j> = 1/20, so about the origin
        I_xx = m (b^2 + c^2) / 10,  I_xy = -m a b / 20.
    """

    A, B, C = 0.02, 0.03, 0.04
    MASS = 0.25

    def _mesh(self, offset: np.ndarray | None = None):  # type: ignore[no-untyped-def]
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [self.A, 0.0, 0.0],
                [0.0, self.B, 0.0],
                [0.0, 0.0, self.C],
            ]
        )
        if offset is not None:
            vertices = vertices + offset
        return tetrahedron_mesh(vertices)

    def _closed_form_about_origin(self) -> np.ndarray:
        a, b, c, m = self.A, self.B, self.C, self.MASS
        return np.array(
            [
                [m * (b * b + c * c) / 10.0, -m * a * b / 20.0, -m * a * c / 20.0],
                [-m * a * b / 20.0, m * (a * a + c * c) / 10.0, -m * b * c / 20.0],
                [-m * a * c / 20.0, -m * b * c / 20.0, m * (a * a + b * b) / 10.0],
            ]
        )

    def test_volume_and_centroid(self) -> None:
        props = compute_mass_properties(self._mesh(), mass_kg=self.MASS)
        assert props.volume_m3 == pytest.approx(
            self.A * self.B * self.C / 6.0, rel=FP_TOL
        )
        np.testing.assert_allclose(
            props.centroid_m,
            np.array([self.A, self.B, self.C]) / 4.0,
            rtol=1e-13,
            atol=1e-18,
        )

    def test_inertia_about_origin(self) -> None:
        props = compute_mass_properties(self._mesh(), mass_kg=self.MASS)
        np.testing.assert_allclose(
            props.inertia_about_origin_kg_m2,
            self._closed_form_about_origin(),
            rtol=1e-11,
            atol=1e-18,
        )

    def test_translated_tetrahedron_still_exact(self) -> None:
        """No vertex at the origin: the signed tetrahedra must cancel."""
        offset = np.array([0.37, -0.21, 0.13])
        props = compute_mass_properties(self._mesh(offset), mass_kg=self.MASS)
        assert props.volume_m3 == pytest.approx(
            self.A * self.B * self.C / 6.0, rel=1e-11
        )
        np.testing.assert_allclose(
            props.centroid_m,
            np.array([self.A, self.B, self.C]) / 4.0 + offset,
            rtol=1e-9,
            atol=1e-15,
        )
        centred = self._closed_form_about_origin() - self.MASS * (
            float(
                np.dot(
                    np.array([self.A, self.B, self.C]) / 4.0,
                    np.array([self.A, self.B, self.C]) / 4.0,
                )
            )
            * np.eye(3)
            - np.outer(
                np.array([self.A, self.B, self.C]) / 4.0,
                np.array([self.A, self.B, self.C]) / 4.0,
            )
        )
        np.testing.assert_allclose(props.inertia_kg_m2, centred, rtol=1e-8, atol=1e-16)


class TestCurvedSolidsConverge:
    def test_sphere_matches_the_closed_form(self) -> None:
        radius, mass = 0.021335, 0.04593  # a golf ball, for good measure
        props = compute_mass_properties(
            icosphere_mesh(radius, subdivisions=5), mass_kg=mass
        )
        assert props.volume_m3 == pytest.approx(4.0 / 3.0 * np.pi * radius**3, rel=1e-3)
        np.testing.assert_allclose(props.centroid_m, np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(
            props.inertia_kg_m2,
            sphere_inertia_about_centroid(mass, radius),
            rtol=2e-3,
            atol=1e-15,
        )

    def test_sphere_volume_error_falls_with_refinement(self) -> None:
        radius = 0.02
        exact = 4.0 / 3.0 * np.pi * radius**3
        errors = [
            abs(
                compute_mass_properties(
                    icosphere_mesh(radius, subdivisions=n), density_kg_m3=1.0
                ).volume_m3
                - exact
            )
            for n in (2, 3, 4)
        ]
        assert errors[1] < errors[0] / 3.0
        assert errors[2] < errors[1] / 3.0

    def test_cylinder_matches_the_closed_form(self) -> None:
        radius, height, mass = 0.015, 0.04, 0.2
        props = compute_mass_properties(
            cylinder_mesh(radius, height, n_segments=512), mass_kg=mass
        )
        assert props.volume_m3 == pytest.approx(np.pi * radius**2 * height, rel=1e-4)
        np.testing.assert_allclose(props.centroid_m, np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(
            props.inertia_kg_m2,
            cylinder_inertia_about_centroid(mass, radius, height),
            rtol=1e-4,
            atol=1e-15,
        )

    def test_cylinder_volume_error_falls_with_refinement(self) -> None:
        radius, height = 0.015, 0.04
        exact = np.pi * radius**2 * height
        errors = [
            abs(
                compute_mass_properties(
                    cylinder_mesh(radius, height, n_segments=n), density_kg_m3=1.0
                ).volume_m3
                - exact
            )
            for n in (16, 32, 64)
        ]
        assert errors[1] < errors[0] / 3.0
        assert errors[2] < errors[1] / 3.0


class TestInertiaAboutAnArbitraryAxis:
    def test_matches_the_principal_value_for_a_principal_axis(self) -> None:
        lx, ly, lz, mass = 0.03, 0.02, 0.01, 0.3
        props = compute_mass_properties(box_mesh(lx, ly, lz), mass_kg=mass)
        moment = props.inertia_about_axis_kg_m2(
            point_m=np.zeros(3), direction=np.array([0.0, 0.0, 1.0])
        )
        assert moment == pytest.approx(mass * (lx**2 + ly**2) / 12.0, rel=1e-11)

    def test_parallel_axis_shift(self) -> None:
        lx, ly, lz, mass = 0.03, 0.02, 0.01, 0.3
        props = compute_mass_properties(box_mesh(lx, ly, lz), mass_kg=mass)
        offset = 0.05
        moment = props.inertia_about_axis_kg_m2(
            point_m=np.array([offset, 0.0, 0.0]), direction=np.array([0.0, 0.0, 1.0])
        )
        assert moment == pytest.approx(
            mass * (lx**2 + ly**2) / 12.0 + mass * offset**2, rel=1e-11
        )

    def test_direction_is_normalised_internally(self) -> None:
        props = compute_mass_properties(box_mesh(0.03, 0.02, 0.01), mass_kg=0.3)
        unit = props.inertia_about_axis_kg_m2(
            point_m=np.zeros(3), direction=np.array([0.0, 0.0, 1.0])
        )
        scaled = props.inertia_about_axis_kg_m2(
            point_m=np.zeros(3), direction=np.array([0.0, 0.0, 7.0])
        )
        assert scaled == pytest.approx(unit, rel=1e-13)

    def test_rejects_a_zero_direction(self) -> None:
        props = compute_mass_properties(box_mesh(0.03, 0.02, 0.01), mass_kg=0.3)
        with pytest.raises(ValueError):
            props.inertia_about_axis_kg_m2(point_m=np.zeros(3), direction=np.zeros(3))


class TestTensorProperties:
    def test_inertia_is_symmetric_and_positive_definite(self) -> None:
        props = compute_mass_properties(box_mesh(0.03, 0.02, 0.01), mass_kg=0.3)
        np.testing.assert_allclose(
            props.inertia_kg_m2, props.inertia_kg_m2.T, rtol=0, atol=1e-20
        )
        assert np.all(np.linalg.eigvalsh(props.inertia_kg_m2) > 0.0)

    def test_triangle_inequality_on_principal_moments(self) -> None:
        props = compute_mass_properties(
            cylinder_mesh(0.015, 0.04, n_segments=64), mass_kg=0.2
        )
        moments = np.sort(np.linalg.eigvalsh(props.inertia_kg_m2))
        assert moments[0] + moments[1] >= moments[2] * (1.0 - 1e-12)

    def test_rotating_the_frame_gives_r_i_rt(self) -> None:
        mesh = box_mesh(0.03, 0.02, 0.01)
        rotation = _rotation(np.array([1.0, 2.0, -0.5]), 0.9)
        base = compute_mass_properties(mesh, mass_kg=0.3)
        turned = compute_mass_properties(
            mesh.transformed(rotation=rotation), mass_kg=0.3
        )
        np.testing.assert_allclose(
            turned.inertia_kg_m2,
            rotation @ base.inertia_kg_m2 @ rotation.T,
            rtol=1e-10,
            atol=1e-18,
        )
        assert turned.volume_m3 == pytest.approx(base.volume_m3, rel=1e-12)
