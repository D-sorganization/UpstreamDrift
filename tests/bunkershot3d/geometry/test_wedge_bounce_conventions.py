"""Bounce-convention typing tests (issue #8609).

The two bounce conventions - the Acushnet patent's *geometric* bounce
(measured to the true trailing contact point, >20 deg) and *marketed*
bounce (measured to the ground-contact plane, 4-14 deg) - must be
impossible to mix, not merely discouraged.
"""

from __future__ import annotations

import math

import pytest

from bunkershot3d.geometry.bounce import (
    BounceConvention,
    GeometricBounce,
    MarketedBounce,
    geometric_from_marketed,
    marketed_from_geometric,
)

pytestmark = pytest.mark.unit

DATUM_M = 1.2e-3


class TestConventionTagging:
    def test_each_convention_carries_its_tag(self) -> None:
        assert GeometricBounce(20.78).convention is BounceConvention.GEOMETRIC
        assert MarketedBounce(10.0).convention is BounceConvention.MARKETED

    def test_radians_accessor(self) -> None:
        assert GeometricBounce(20.0).angle_rad == pytest.approx(math.radians(20.0))

    def test_distinct_types(self) -> None:
        assert type(GeometricBounce(10.0)) is not type(MarketedBounce(10.0))


class TestMixingIsImpossible:
    def test_equal_angles_in_different_conventions_are_not_equal(self) -> None:
        assert GeometricBounce(10.0) != MarketedBounce(10.0)

    def test_addition_across_conventions_raises(self) -> None:
        with pytest.raises(TypeError):
            _ = GeometricBounce(20.0) + MarketedBounce(4.0)  # type: ignore[operator]

    def test_subtraction_across_conventions_raises(self) -> None:
        with pytest.raises(TypeError):
            _ = MarketedBounce(10.0) - GeometricBounce(20.0)  # type: ignore[operator]

    def test_addition_within_a_convention_is_type_preserving(self) -> None:
        total = GeometricBounce(20.0) + GeometricBounce(1.0)
        assert isinstance(total, GeometricBounce)
        assert total.angle_deg == pytest.approx(21.0)

    def test_shifted_by_preserves_the_concrete_type(self) -> None:
        for bounce in (GeometricBounce(20.0), MarketedBounce(10.0)):
            shifted = bounce.shifted_by(-3.0)
            assert type(shifted) is type(bounce)
            assert shifted.angle_deg == pytest.approx(bounce.angle_deg - 3.0)

    def test_bounce_is_immutable(self) -> None:
        bounce = GeometricBounce(20.0)
        with pytest.raises((AttributeError, TypeError)):
            bounce.angle_deg = 4.0  # type: ignore[misc]


class TestValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 95.0, -95.0])
    def test_rejects_non_physical_angles(self, bad: float) -> None:
        with pytest.raises(ValueError):
            GeometricBounce(bad)

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(TypeError):
            GeometricBounce("20")  # type: ignore[arg-type]


class TestConversion:
    """Conversion is explicit and needs the datum geometry, never implicit."""

    def test_marketed_is_lower_than_geometric_for_a_relieved_leading_edge(
        self,
    ) -> None:
        geometric = GeometricBounce(21.0)
        marketed = marketed_from_geometric(
            geometric,
            sole_width_m=18.0e-3,
            entry_height_m=3.0e-3,
            datum_offset_m=DATUM_M,
        )
        assert isinstance(marketed, MarketedBounce)
        assert 4.0 < marketed.angle_deg < 14.0
        assert marketed.angle_deg < geometric.angle_deg

    def test_hand_computed_value(self) -> None:
        # h_T = 18 mm * tan(21 deg) = 6.9084 mm; the effective sole line runs
        # from the 1.2 mm datum (3 mm below the leading-edge point) to the
        # trailing contact point: atan((6.9084 - 3) / (18 - 1.2)).
        expected = math.degrees(
            math.atan2(18.0 * math.tan(math.radians(21.0)) - 3.0, 18.0 - 1.2)
        )
        marketed = marketed_from_geometric(
            GeometricBounce(21.0),
            sole_width_m=18.0e-3,
            entry_height_m=3.0e-3,
            datum_offset_m=DATUM_M,
        )
        assert marketed.angle_deg == pytest.approx(expected, abs=1e-12)

    def test_round_trip(self) -> None:
        geometric = GeometricBounce(18.42)
        marketed = marketed_from_geometric(
            geometric,
            sole_width_m=16.0e-3,
            entry_height_m=2.5e-3,
            datum_offset_m=DATUM_M,
        )
        back = geometric_from_marketed(
            marketed,
            sole_width_m=16.0e-3,
            entry_height_m=2.5e-3,
            datum_offset_m=DATUM_M,
        )
        assert isinstance(back, GeometricBounce)
        assert back.angle_deg == pytest.approx(geometric.angle_deg, abs=1e-10)

    def test_deep_leading_edge_relief_can_null_the_marketed_bounce(self) -> None:
        marketed = marketed_from_geometric(
            GeometricBounce(21.0),
            sole_width_m=18.0e-3,
            entry_height_m=6.91e-3,
            datum_offset_m=DATUM_M,
        )
        assert marketed.angle_deg == pytest.approx(0.0, abs=0.1)

    def test_conversion_rejects_the_wrong_convention(self) -> None:
        with pytest.raises(TypeError):
            marketed_from_geometric(
                MarketedBounce(10.0),  # type: ignore[arg-type]
                sole_width_m=18.0e-3,
                entry_height_m=3.0e-3,
                datum_offset_m=DATUM_M,
            )

    def test_conversion_rejects_a_datum_wider_than_the_sole(self) -> None:
        with pytest.raises(ValueError):
            marketed_from_geometric(
                GeometricBounce(20.0),
                sole_width_m=1.0e-3,
                entry_height_m=3.0e-3,
                datum_offset_m=DATUM_M,
            )
