"""Lofted wedge-mesh tests (issue #8609).

Replaces finding B20: the old `ClubheadGenerator` emitted a 6-vertex
triangular prism with a hard-coded 20 mm sole and no camber, relief,
rocker, lie or mass properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.geometry.lofting import (
    build_wedge_mesh,
    shaft_axis,
    wedge_mass_properties,
)
from bunkershot3d.geometry.mesh import TriangleMesh, check_mesh_validity
from bunkershot3d.geometry.wedge import WedgeGeometry

from .conftest import build_reference_wedge

pytestmark = pytest.mark.unit


class TestMeshIsAVerifiedSolid:
    def test_watertight_manifold_and_outward(self, wedge_mesh: TriangleMesh) -> None:
        report = check_mesh_validity(wedge_mesh)
        assert report.is_watertight_solid
        assert report.euler_characteristic == 2
        assert report.genus == 0
        assert report.signed_volume_m3 > 0.0
        assert report.n_degenerate_faces == 0
        assert report.n_unreferenced_vertices == 0

    def test_resolution_is_configurable_and_still_valid(
        self, wedge: WedgeGeometry
    ) -> None:
        mesh = build_wedge_mesh(wedge, n_profile_points=24, n_stations=9)
        assert check_mesh_validity(mesh).is_watertight_solid
        assert mesh.n_faces < build_wedge_mesh(wedge).n_faces

    def test_rejects_absurd_resolutions(self, wedge: WedgeGeometry) -> None:
        with pytest.raises(ValueError):
            build_wedge_mesh(wedge, n_stations=1)
        with pytest.raises(ValueError):
            build_wedge_mesh(wedge, n_profile_points=3)


class TestGeometryIsActuallyRepresented:
    def test_spans_the_blade_length(self, wedge_mesh: TriangleMesh) -> None:
        span = float(wedge_mesh.vertices[:, 1].max() - wedge_mesh.vertices[:, 1].min())
        assert span == pytest.approx(0.078, rel=1e-9)

    def test_leading_edge_rises_toward_heel_and_toe(
        self, wedge_mesh: TriangleMesh
    ) -> None:
        vertices = np.asarray(wedge_mesh.vertices)
        leading = vertices[np.isclose(vertices[:, 0], vertices[:, 0].min(), atol=1e-9)]
        assert leading.shape[0] >= 3
        centre = leading[np.argmin(np.abs(leading[:, 1]))]
        heel = leading[np.argmin(leading[:, 1])]
        toe = leading[np.argmax(leading[:, 1])]
        assert heel[2] > centre[2]
        assert toe[2] > centre[2]

    def test_heel_relief_is_stronger_than_toe_relief(self) -> None:
        geometry = build_reference_wedge(
            heel_relief_fraction=0.30, toe_relief_fraction=0.05
        )
        mesh = build_wedge_mesh(geometry)
        vertices = np.asarray(mesh.vertices)
        heel = vertices[vertices[:, 1] < vertices[:, 1].min() + 1e-9]
        toe = vertices[vertices[:, 1] > vertices[:, 1].max() - 1e-9]
        assert heel[:, 2].min() > toe[:, 2].min()

    def test_trailing_relief_removes_material(self) -> None:
        stock = wedge_mass_properties(
            build_reference_wedge(trailing_relief_fraction=0.0)
        )
        ground = wedge_mass_properties(
            build_reference_wedge(trailing_relief_fraction=0.35)
        )
        assert ground.volume_m3 < stock.volume_m3

    def test_volume_scales_with_blade_length(self) -> None:
        short = wedge_mass_properties(build_reference_wedge(blade_length_mm=70.0))
        long_head = wedge_mass_properties(build_reference_wedge(blade_length_mm=84.0))
        assert long_head.volume_m3 > short.volume_m3

    def test_bounce_changes_the_sole(self) -> None:
        from bunkershot3d.geometry.bounce import GeometricBounce
        from bunkershot3d.geometry.lofting import CamberFit

        # The band of realisable camber areas climbs with bounce, so the
        # reference wedge's 55 mm^2 is not constructible at 16 deg. That is
        # the point of #8698: ask for the nearest constructible sole
        # explicitly rather than have one substituted behind your back.
        low = build_wedge_mesh(
            build_reference_wedge(geometric_bounce=GeometricBounce(16.0)),
            camber_fit=CamberFit.NEAREST,
        )
        high = build_wedge_mesh(
            build_reference_wedge(geometric_bounce=GeometricBounce(24.0)),
            camber_fit=CamberFit.NEAREST,
        )
        assert float(low.vertices[:, 2].min()) > float(high.vertices[:, 2].min())

    def test_face_progression_shifts_the_head(self) -> None:
        base = build_wedge_mesh(build_reference_wedge(face_progression_mm=0.0))
        forward = build_wedge_mesh(build_reference_wedge(face_progression_mm=4.0))
        assert float(forward.vertices[:, 0].min()) == pytest.approx(
            float(base.vertices[:, 0].min()) + 0.004, abs=1e-12
        )


class TestWedgeMassProperties:
    def test_head_mass_is_respected(self, wedge: WedgeGeometry) -> None:
        props = wedge_mass_properties(wedge)
        assert props.mass_kg == pytest.approx(wedge.head_mass_kg)
        assert props.volume_m3 > 0.0
        assert 5_000.0 < props.density_kg_m3 < 15_000.0

    def test_centre_of_gravity_sits_above_the_leading_edge(
        self, wedge: WedgeGeometry
    ) -> None:
        # Patent band for CG height above the leading edge is 9.65-17.02 mm;
        # this parametric blade is a first-cut shape, so the assertion is a
        # plausibility band, not a validated match.
        props = wedge_mass_properties(wedge)
        assert 0.004 < float(props.centroid_m[2]) < 0.025

    def test_inertia_about_the_shaft_axis_is_positive_and_plausible(
        self, wedge: WedgeGeometry
    ) -> None:
        point, direction = shaft_axis(wedge)
        props = wedge_mass_properties(wedge)
        moment = props.inertia_about_axis_kg_m2(point_m=point, direction=direction)
        assert moment > 0.0
        # 2200-3200 g.cm^2 published for irons == 2.2e-4 to 3.2e-4 kg.m^2.
        assert 5e-5 < moment < 2e-3

    def test_shaft_axis_direction_follows_the_lie_angle(
        self, wedge: WedgeGeometry
    ) -> None:
        _, direction = shaft_axis(wedge)
        assert float(np.linalg.norm(direction)) == pytest.approx(1.0, abs=1e-12)
        assert float(direction[2]) == pytest.approx(np.sin(wedge.lie_rad), abs=1e-12)
        assert float(direction[0]) == pytest.approx(0.0, abs=1e-12)

    def test_moi_about_the_shaft_axis_grows_with_blade_length(self) -> None:
        short = build_reference_wedge(blade_length_mm=70.0)
        long_head = build_reference_wedge(blade_length_mm=86.0)
        short_moment = wedge_mass_properties(short).inertia_about_axis_kg_m2(
            *_axis_args(short)
        )
        long_moment = wedge_mass_properties(long_head).inertia_about_axis_kg_m2(
            *_axis_args(long_head)
        )
        assert long_moment > short_moment


def _axis_args(geometry: WedgeGeometry) -> tuple[np.ndarray, np.ndarray]:
    point, direction = shaft_axis(geometry)
    return point, direction
