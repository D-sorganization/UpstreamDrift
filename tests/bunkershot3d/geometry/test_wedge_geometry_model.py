"""`WedgeGeometry` value-object tests (issue #8609).

Covers the Acushnet sole schema (US10143900B2 / US10661131B2): sole width
d1, the 1.2 mm datum d2, sole entry height d3, entry angle Phi, sole radii
rho1/rho2, camber area, and the derived Sole Contour Ratio and
camber-to-bounce area ratio.
"""

from __future__ import annotations

import math

import pytest

from bunkershot3d.geometry.bounce import (
    GeometricBounce,
    MarketedBounce,
)
from bunkershot3d.geometry.wedge import PatentBand, WedgeGeometry

from .conftest import build_reference_wedge

pytestmark = pytest.mark.unit


class TestValueObject:
    def test_is_frozen(self, wedge: WedgeGeometry) -> None:
        with pytest.raises((AttributeError, TypeError)):
            wedge.loft_deg = 60.0  # type: ignore[misc]

    def test_si_units_internally(self, wedge: WedgeGeometry) -> None:
        assert wedge.sole_width_m == pytest.approx(21.0e-3)
        assert wedge.datum_offset_m == pytest.approx(1.2e-3)
        assert wedge.head_mass_kg == pytest.approx(0.304)

    def test_datum_is_the_patent_constant(self, wedge: WedgeGeometry) -> None:
        assert wedge.datum_offset_m == pytest.approx(1.2e-3)

    def test_angle_accessors_are_suffixed_and_consistent(
        self, wedge: WedgeGeometry
    ) -> None:
        assert wedge.loft_rad == pytest.approx(math.radians(wedge.loft_deg))
        assert wedge.lie_rad == pytest.approx(math.radians(wedge.lie_deg))

    def test_equality_is_by_value(self) -> None:
        assert build_reference_wedge() == build_reference_wedge()
        assert build_reference_wedge() != build_reference_wedge(loft_deg=58.0)


class TestDerivedQuantities:
    def test_sole_entry_angle_is_derived_from_the_datum(
        self, wedge: WedgeGeometry
    ) -> None:
        expected = math.degrees(math.atan2(3.5, 1.2))
        assert wedge.sole_entry_angle_deg == pytest.approx(expected)
        assert wedge.sole_entry_angle_deg > 67.5  # most-preferred band

    def test_sole_contour_ratio(self, wedge: WedgeGeometry) -> None:
        assert wedge.sole_contour_ratio == pytest.approx(7.5 / 42.0)
        assert wedge.sole_contour_ratio < 0.25

    def test_camber_to_bounce_area_ratio(self, wedge: WedgeGeometry) -> None:
        assert wedge.camber_to_bounce_ratio_mm2_per_deg == pytest.approx(55.0 / 21.0)
        assert wedge.camber_to_bounce_ratio_mm2_per_deg > 2.0

    def test_trailing_contact_drop(self, wedge: WedgeGeometry) -> None:
        expected = 21.0e-3 * math.tan(math.radians(21.0))
        assert wedge.trailing_contact_drop_m == pytest.approx(expected)

    def test_marketed_bounce_is_derived_and_tagged(self, wedge: WedgeGeometry) -> None:
        marketed = wedge.marketed_bounce
        assert isinstance(marketed, MarketedBounce)
        assert 4.0 < marketed.angle_deg < 14.0  # published wedges sit in 4-14 deg

    def test_geometric_bounce_is_tagged(self, wedge: WedgeGeometry) -> None:
        assert isinstance(wedge.geometric_bounce, GeometricBounce)


class TestInvariants:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("loft_deg", 0.0),
            ("loft_deg", 90.0),
            ("loft_deg", float("nan")),
            ("lie_deg", 0.0),
            ("lie_deg", 100.0),
            ("sole_width_mm", 0.0),
            ("sole_width_mm", -1.0),
            ("entry_height_mm", 0.0),
            ("leading_edge_radius_mm", 0.0),
            ("trailing_edge_radius_mm", 0.0),
            ("sole_camber_area_mm2", 0.0),
            ("blade_length_mm", 0.0),
            ("face_height_mm", 0.0),
            ("topline_width_mm", 0.0),
            ("head_mass_g", 0.0),
            ("head_mass_g", 5000.0),
            ("heel_relief_fraction", 1.0),
            ("toe_relief_fraction", -0.1),
            ("trailing_relief_fraction", 1.5),
        ],
    )
    def test_rejects_non_physical_parameters(self, field: str, value: float) -> None:
        with pytest.raises(ValueError):
            build_reference_wedge(**{field: value})

    def test_rejects_a_datum_wider_than_the_sole(self) -> None:
        with pytest.raises(ValueError):
            build_reference_wedge(sole_width_mm=1.0)

    def test_rejects_a_leading_radius_larger_than_the_trailing_radius(self) -> None:
        with pytest.raises(ValueError):
            build_reference_wedge(
                leading_edge_radius_mm=50.0, trailing_edge_radius_mm=42.0
            )

    def test_rejects_an_entry_shallower_than_the_bounce_chord(self) -> None:
        # The sole must fall below the leading-edge/trailing-contact chord at
        # the datum, otherwise the profile is concave and the camber area is
        # not defined.
        with pytest.raises(ValueError):
            build_reference_wedge(entry_height_mm=0.2)

    def test_rejects_a_wrongly_typed_bounce(self) -> None:
        with pytest.raises(TypeError):
            build_reference_wedge(geometric_bounce=MarketedBounce(10.0))

    def test_invariant_violations_raise_not_assert(self) -> None:
        # `python -O` strips `assert`; safety-critical validation must raise.
        source = __import__("bunkershot3d.geometry.wedge", fromlist=["wedge"]).__file__
        assert source is not None
        with open(source, encoding="utf-8") as handle:
            body = handle.read()
        assert "\n        assert " not in body
        assert "\n    assert " not in body


class TestPatentCompliance:
    def test_reference_wedge_is_in_the_most_preferred_bands(
        self, wedge: WedgeGeometry
    ) -> None:
        report = wedge.patent_compliance()
        assert report["sole_width"] is PatentBand.MOST_PREFERRED
        assert report["entry_height"] is PatentBand.MOST_PREFERRED
        assert report["sole_entry_angle"] is PatentBand.MOST_PREFERRED
        assert report["leading_edge_radius"] is PatentBand.MOST_PREFERRED
        assert report["trailing_edge_radius"] is PatentBand.MOST_PREFERRED
        assert report["sole_camber_area"] is PatentBand.MOST_PREFERRED
        assert report["sole_contour_ratio"] is PatentBand.MOST_PREFERRED
        assert report["geometric_bounce"] is PatentBand.MOST_PREFERRED
        # The camber-to-bounce ratio's most-preferred band (>3 mm^2/deg) needs
        # extreme leading-edge relief: the camber area is bounded above by
        # d1^2 tan(theta) / 2 = 85 mm^2 for this sole, and far more tightly by
        # the requirement that the sole stay convex and monotone, so the
        # patent's own worked examples sit near this boundary too.
        assert report["camber_to_bounce_ratio"] is PatentBand.PREFERRED

    def test_out_of_range_is_reported_not_raised(self) -> None:
        narrow = build_reference_wedge(
            sole_width_mm=4.0,
            entry_height_mm=1.2,
            sole_camber_area_mm2=10.0,
            heel_relief_fraction=0.0,
            toe_relief_fraction=0.0,
        )
        report = narrow.patent_compliance()
        assert report["sole_width"] is PatentBand.OUT_OF_RANGE
        assert report["sole_camber_area"] is PatentBand.OUT_OF_RANGE

    def test_broad_band_is_distinguished_from_preferred(self) -> None:
        broad = build_reference_wedge(
            sole_width_mm=7.0,
            entry_height_mm=2.0,
            heel_relief_fraction=0.0,
            toe_relief_fraction=0.0,
        )
        assert broad.patent_compliance()["sole_width"] is PatentBand.BROAD

    def test_report_covers_every_schema_parameter(self, wedge: WedgeGeometry) -> None:
        assert set(wedge.patent_compliance()) == set(WedgeGeometry.patent_parameters())
