"""Parametric sole cross-section tests (issue #8609).

All measurements follow the Acushnet convention: a vertical plane
perpendicular to the leading edge, x rearward from the leading-edge
point, z up.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.geometry.profile import (
    build_section_polygon,
    build_sole_profile,
    polygon_area_m2,
)
from bunkershot3d.geometry.wedge import WedgeGeometry

from .conftest import build_reference_wedge

pytestmark = pytest.mark.unit


def _cross_2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _circumradius(points: np.ndarray) -> float:
    a, b, c = points
    side_a = float(np.linalg.norm(b - c))
    side_b = float(np.linalg.norm(a - c))
    side_c = float(np.linalg.norm(a - b))
    area = 0.5 * abs(float(_cross_2d(b - a, c - a)))
    return side_a * side_b * side_c / (4.0 * area)


class TestSoleProfile:
    def test_starts_at_the_leading_edge_point(self, wedge: WedgeGeometry) -> None:
        profile = build_sole_profile(wedge)
        np.testing.assert_allclose(profile.points_m[0], np.zeros(2), atol=1e-18)

    def test_ends_at_the_trailing_contact_point(self, wedge: WedgeGeometry) -> None:
        profile = build_sole_profile(wedge)
        np.testing.assert_allclose(
            profile.points_m[-1],
            np.array([wedge.sole_width_m, -wedge.trailing_contact_drop_m]),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_passes_through_the_1_2_mm_datum(self, wedge: WedgeGeometry) -> None:
        profile = build_sole_profile(wedge)
        z_at_datum = profile.height_at_m(wedge.datum_offset_m)
        assert z_at_datum == pytest.approx(-wedge.entry_height_m, abs=1e-12)

    def test_entry_angle_matches_the_schema(self, wedge: WedgeGeometry) -> None:
        profile = build_sole_profile(wedge)
        chord_angle = math.degrees(
            math.atan2(-profile.height_at_m(wedge.datum_offset_m), wedge.datum_offset_m)
        )
        assert chord_angle == pytest.approx(wedge.sole_entry_angle_deg, abs=1e-9)

    def test_descends_monotonically_to_the_trailing_contact(
        self, wedge: WedgeGeometry
    ) -> None:
        profile = build_sole_profile(wedge)
        drops = np.diff(profile.points_m[:, 1])
        assert np.all(drops <= 1e-15)
        assert np.all(np.diff(profile.points_m[:, 0]) > 0.0)

    def test_never_dips_below_the_ground_plane(self, wedge: WedgeGeometry) -> None:
        profile = build_sole_profile(wedge)
        ground = -wedge.trailing_contact_drop_m
        assert float(profile.points_m[:, 1].min()) >= ground - 1e-15

    def test_camber_area_matches_the_declared_value(self, wedge: WedgeGeometry) -> None:
        profile = build_sole_profile(wedge)
        assert profile.camber_area_m2 == pytest.approx(
            wedge.sole_camber_area_m2, rel=1e-10
        )

    @pytest.mark.parametrize("camber_mm2", [50.0, 55.0, 60.0])
    def test_camber_area_is_honoured_across_the_patent_band(
        self, camber_mm2: float
    ) -> None:
        geometry = build_reference_wedge(sole_camber_area_mm2=camber_mm2)
        profile = build_sole_profile(geometry)
        assert profile.camber_area_m2 * 1e6 == pytest.approx(camber_mm2, rel=1e-10)

    def test_leading_arc_realises_the_requested_radius(
        self, wedge: WedgeGeometry
    ) -> None:
        profile = build_sole_profile(wedge)
        assert profile.leading_arc_radius_m == pytest.approx(
            wedge.leading_edge_radius_m, rel=1e-12
        )
        arc = profile.points_m[:3]
        assert _circumradius(arc) == pytest.approx(
            wedge.leading_edge_radius_m, rel=1e-6
        )

    def test_trailing_arc_realises_the_requested_radius(
        self, wedge: WedgeGeometry
    ) -> None:
        profile = build_sole_profile(wedge)
        assert profile.trailing_arc_radius_m == pytest.approx(
            wedge.trailing_edge_radius_m, rel=1e-12
        )
        arc = profile.points_m[-3:]
        assert _circumradius(arc) == pytest.approx(
            wedge.trailing_edge_radius_m, rel=1e-4
        )

    def test_sole_is_tangent_to_the_ground_at_the_trailing_contact(
        self, wedge: WedgeGeometry
    ) -> None:
        profile = build_sole_profile(wedge)
        last = profile.points_m[-1] - profile.points_m[-2]
        assert abs(math.degrees(math.atan2(-last[1], last[0]))) < 0.5

    def test_sampling_resolution_is_configurable(self, wedge: WedgeGeometry) -> None:
        coarse = build_sole_profile(wedge, n_points=17)
        fine = build_sole_profile(wedge, n_points=65)
        assert coarse.points_m.shape[0] == 17
        assert fine.points_m.shape[0] == 65
        assert fine.camber_area_m2 == pytest.approx(
            wedge.sole_camber_area_m2, rel=1e-12
        )


class TestInfeasibleRequests:
    def test_leading_radius_too_small_for_the_entry_chord_raises(self) -> None:
        geometry = build_reference_wedge(
            leading_edge_radius_mm=1.0, trailing_edge_radius_mm=42.0
        )
        with pytest.raises(ValueError, match="leading"):
            build_sole_profile(geometry)

    def test_impossible_camber_area_raises(self) -> None:
        geometry = build_reference_wedge(sole_camber_area_mm2=500.0)
        with pytest.raises(ValueError):
            build_sole_profile(geometry)

    def test_error_is_a_raise_not_an_assert(self) -> None:
        module = __import__("bunkershot3d.geometry.profile", fromlist=["profile"])
        assert module.__file__ is not None
        with open(module.__file__, encoding="utf-8") as handle:
            body = handle.read()
        assert "\n    assert " not in body
        assert "\n        assert " not in body


class TestSectionPolygon:
    def test_is_closed_and_counter_clockwise(self, wedge: WedgeGeometry) -> None:
        polygon = build_section_polygon(wedge)
        assert polygon.shape[1] == 2
        assert not np.allclose(polygon[0], polygon[-1])  # implicit closure
        assert polygon_area_m2(polygon) > 0.0

    def test_is_convex(self, wedge: WedgeGeometry) -> None:
        polygon = build_section_polygon(wedge)
        edges = np.roll(polygon, -1, axis=0) - polygon
        crosses = _cross_2d(edges, np.roll(edges, -1, axis=0))
        assert np.all(crosses >= -1e-15)

    def test_contains_the_leading_edge_and_trailing_contact(
        self, wedge: WedgeGeometry
    ) -> None:
        polygon = build_section_polygon(wedge)
        assert np.isclose(polygon, np.zeros(2)).all(axis=1).any()
        trailing = np.array([wedge.sole_width_m, -wedge.trailing_contact_drop_m])
        assert np.isclose(polygon, trailing, atol=1e-12).all(axis=1).any()

    def test_face_top_is_set_by_loft_and_face_height(
        self, wedge: WedgeGeometry
    ) -> None:
        polygon = build_section_polygon(wedge)
        expected = np.array(
            [
                wedge.face_height_m * math.sin(wedge.loft_rad),
                wedge.face_height_m * math.cos(wedge.loft_rad),
            ]
        )
        assert np.isclose(polygon, expected, atol=1e-12).all(axis=1).any()

    def test_area_grows_with_face_height(self) -> None:
        small = polygon_area_m2(build_section_polygon(build_reference_wedge()))
        large = polygon_area_m2(
            build_section_polygon(build_reference_wedge(face_height_mm=44.0))
        )
        assert large > small
