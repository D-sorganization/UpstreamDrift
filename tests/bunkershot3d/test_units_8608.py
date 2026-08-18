"""Unit-convention discipline for BunkerShot3D (issue #8608, W1).

Two things are pinned here:

1. **Every** conversion in :data:`bunkershot3d.units.CONVERSIONS` round-trips.
   The registry is iterated rather than enumerated, so a conversion added
   without a round-trip property test cannot exist.
2. Every field of every domain value object is either explicitly dimensionless
   or carries a recognised SI suffix. The convention is machine-checked, not
   just documented in a docstring.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bunkershot3d import units
from bunkershot3d.domain import (
    ContactMaterial,
    DomainBox,
    GrainPopulation,
    SolverSettings,
    SwingCondition,
    TrajectorySource,
)
from bunkershot3d.exceptions import UnitConventionError, UnitConversionError

pytestmark = pytest.mark.unit

_ANY = st.floats(
    min_value=-1.0e6, max_value=1.0e6, allow_nan=False, allow_infinity=False
)
_POSITIVE = st.floats(
    min_value=1.0e-6, max_value=1.0e6, allow_nan=False, allow_infinity=False
)


def _strategy_for(conversion: units.Conversion) -> st.SearchStrategy[float]:
    return _POSITIVE if conversion.positive_only else _ANY


class TestRegistryCompleteness:
    """The registry is the single list of conversions this package offers."""

    def test_registry_is_not_empty(self) -> None:
        assert units.CONVERSIONS

    def test_every_public_conversion_function_is_registered(self) -> None:
        """No conversion may be exported without appearing in the registry."""
        registered = {
            function.__name__
            for conversion in units.CONVERSIONS.values()
            for function in (conversion.to_si, conversion.from_si)
        }
        exported = {
            name
            for name in units.__all__
            if callable(getattr(units, name, None))
            and ("_to_" in name)
            and not name.startswith("_")
        }
        assert exported <= registered, sorted(exported - registered)

    def test_every_conversion_declares_both_units(self) -> None:
        for name, conversion in units.CONVERSIONS.items():
            assert conversion.si_unit, name
            assert conversion.other_unit, name
            assert conversion.si_unit != conversion.other_unit, name


@pytest.mark.parametrize("name", sorted(units.CONVERSIONS))
def test_conversion_round_trips(name: str) -> None:
    """``from_si(to_si(x)) == x`` for every registered conversion."""
    conversion = units.CONVERSIONS[name]

    @given(_strategy_for(conversion))
    def check(value: float) -> None:
        recovered = conversion.from_si(conversion.to_si(value))
        assert recovered == pytest.approx(value, rel=1e-12, abs=1e-12)

    check()


@pytest.mark.parametrize("name", sorted(units.CONVERSIONS))
def test_conversion_round_trips_the_other_way(name: str) -> None:
    """``to_si(from_si(x)) == x`` — the inverse direction is also exact."""
    conversion = units.CONVERSIONS[name]

    @given(_strategy_for(conversion))
    def check(value: float) -> None:
        recovered = conversion.to_si(conversion.from_si(value))
        assert recovered == pytest.approx(value, rel=1e-12, abs=1e-12)

    check()


@pytest.mark.parametrize("name", sorted(units.CONVERSIONS))
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_conversions_refuse_non_finite_input(name: str, bad: float) -> None:
    conversion = units.CONVERSIONS[name]
    with pytest.raises(UnitConversionError):
        conversion.to_si(bad)
    with pytest.raises(UnitConversionError):
        conversion.from_si(bad)


class TestReciprocalConversion:
    """Rate/period is the one non-linear conversion and needs its own guard."""

    def test_zero_rate_is_refused(self) -> None:
        with pytest.raises(UnitConversionError):
            units.hz_to_period_s(0.0)

    def test_negative_rate_is_refused(self) -> None:
        with pytest.raises(UnitConversionError):
            units.hz_to_period_s(-1.0)

    def test_one_kilohertz_is_one_millisecond(self) -> None:
        assert units.hz_to_period_s(1000.0) == pytest.approx(1.0e-3)


class TestKnownValues:
    """Spot values, so a sign or factor error cannot hide behind a round trip."""

    def test_ninety_degrees_is_half_pi(self) -> None:
        assert units.deg_to_rad(90.0) == pytest.approx(math.pi / 2.0)

    def test_one_millimetre_is_one_thousandth_of_a_metre(self) -> None:
        assert units.mm_to_m(1.0) == pytest.approx(1.0e-3)

    def test_one_square_millimetre_is_one_millionth_of_a_square_metre(self) -> None:
        assert units.mm2_to_m2(1.0) == pytest.approx(1.0e-6)

    def test_one_gram_is_one_thousandth_of_a_kilogram(self) -> None:
        assert units.g_to_kg(1.0) == pytest.approx(1.0e-3)

    def test_one_kilopascal_is_one_thousand_pascals(self) -> None:
        assert units.kpa_to_pa(1.0) == pytest.approx(1000.0)

    def test_one_mile_per_hour_is_the_exact_si_definition(self) -> None:
        # 1 mile = 1609.344 m exactly, by international agreement.
        assert units.mph_to_mps(1.0) == pytest.approx(1609.344 / 3600.0)


class TestSuffixConvention:
    """``si_unit_for`` is what gives the naming convention teeth."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("sole_width_m", "m"),
            ("plan_area_m2", "m^2"),
            ("bulk_volume_m3", "m^3"),
            ("head_mass_kg", "kg"),
            ("density_kg_m3", "kg/m^3"),
            ("duration_s", "s"),
            ("theta_rad", "rad"),
            ("loft_deg", "deg"),
            ("firmness_pa", "Pa"),
            ("output_rate_hz", "Hz"),
            ("clubhead_speed_mps", "m/s"),
        ],
    )
    def test_recognised_suffixes(self, name: str, expected: str) -> None:
        assert units.si_unit_for(name) == expected

    def test_unsuffixed_physical_name_is_refused(self) -> None:
        with pytest.raises(UnitConventionError, match="width"):
            units.si_unit_for("width")

    def test_longest_suffix_wins(self) -> None:
        """``_kg_m3`` must not be read as ``_m3``."""
        assert units.si_unit_for("grain_density_kg_m3") == "kg/m^3"


#: Field names that carry no dimension. Every other field of every domain
#: value object must be unit-suffixed.
DIMENSIONLESS_FIELDS = frozenset(
    {
        "boundary",
        "coarse_graining_factor",
        "count",
        "delivery",
        "diameter_sigma_log",
        "downsample_grains",
        "file",
        "friction",
        "poisson_ratio",
        "restitution",
    }
)

_VALUE_OBJECTS = (
    ContactMaterial,
    DomainBox,
    GrainPopulation,
    SolverSettings,
    SwingCondition,
    TrajectorySource,
)


@pytest.mark.parametrize("value_object", _VALUE_OBJECTS, ids=lambda c: c.__name__)
def test_every_domain_field_is_dimensionless_or_unit_suffixed(
    value_object: type,
) -> None:

    for field in dataclasses.fields(value_object):
        if field.name in DIMENSIONLESS_FIELDS:
            continue
        # Raises UnitConventionError if the name carries no known suffix.
        assert units.si_unit_for(field.name)
