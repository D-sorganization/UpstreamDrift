"""A clamped sole camber must be detectable by its caller (issue #8698).

``build_wedge_mesh`` substitutes the nearest constructible camber area when
the declared one is outside the band a convex, monotone sole of that width
admits.  The substitution is physically right -- a narrow sole cannot host an
arbitrarily large camber -- but before this issue it was *unobservable*:

* :func:`constructible_camber_range_m2` was not re-exported, so no caller
  could pre-check a design vector;
* no result object carried the effective camber back, so a sweep logged the
  value it *declared* while the mesh carried a different one.

Measured on a 77-point demo sweep, the clamp fired on 40 points and moved the
effective camber over 24.5-61.6 mm^2 against a constant declared 48.0 mm^2 --
enough for a Morris or Sobol' run to attribute variance to a factor the user
believed was pinned.

These tests pin the three halves of the fix: the effective value is on the
result, the band is a public query, and silence is opt-in.
"""

from __future__ import annotations

import dataclasses

import pytest

import bunkershot3d
from bunkershot3d.geometry import (
    CamberFit,
    InconstructibleCamberError,
    LoftedWedge,
    StationCamber,
    constructible_camber_range_m2,
    loft_wedge,
)
from bunkershot3d.geometry.bounce import GeometricBounce
from bunkershot3d.geometry.lofting import build_wedge_mesh
from bunkershot3d.geometry.mesh import check_mesh_validity
from bunkershot3d.geometry.presets import get_preset, preset_names
from bunkershot3d.geometry.wedge import WedgeGeometry

from .conftest import build_reference_wedge

pytestmark = pytest.mark.unit

#: Cheap lofting resolution: these tests are about bookkeeping, not meshes.
COARSE = {"n_profile_points": 24, "n_stations": 5}


def _outside_band_wedge() -> WedgeGeometry:
    """A reference wedge whose declared camber its own sole cannot carry."""
    return build_reference_wedge(sole_camber_area_mm2=45.0)


class TestTheBandIsAPublicQuery:
    """Deliverable 2: a design space can be validated before it is sampled."""

    def test_re_exported_from_the_geometry_package(self) -> None:
        assert "constructible_camber_range_m2" in bunkershot3d.geometry.__all__

    def test_re_exported_from_the_top_level(self) -> None:
        assert "constructible_camber_range_m2" in bunkershot3d.__all__
        assert (
            bunkershot3d.constructible_camber_range_m2
            is bunkershot3d.geometry.constructible_camber_range_m2
        )

    def test_the_band_brackets_a_constructible_declaration(self) -> None:
        wedge = build_reference_wedge()
        low, high = constructible_camber_range_m2(wedge, n_points=40)
        assert low < wedge.sole_camber_area_m2 < high

    def test_the_band_excludes_an_inconstructible_declaration(self) -> None:
        wedge = _outside_band_wedge()
        low, high = constructible_camber_range_m2(wedge, n_points=40)
        assert not low <= wedge.sole_camber_area_m2 <= high


class TestSilenceIsOptIn:
    """Deliverable 3: the default refuses; nearest-constructible is asked for."""

    def test_a_declared_camber_outside_the_band_raises_by_default(self) -> None:
        with pytest.raises(InconstructibleCamberError):
            build_wedge_mesh(_outside_band_wedge(), **COARSE)

    def test_the_refusal_is_also_a_value_error(self) -> None:
        """Callers written before this issue catch ``ValueError``."""
        with pytest.raises(ValueError):
            loft_wedge(_outside_band_wedge(), **COARSE)

    def test_the_refusal_quotes_the_request_and_the_band(self) -> None:
        wedge = _outside_band_wedge()
        with pytest.raises(InconstructibleCamberError) as caught:
            loft_wedge(wedge, **COARSE)
        error = caught.value
        assert error.requested_camber_area_m2 == wedge.sole_camber_area_m2
        low, high = error.constructible_range_m2
        assert low > wedge.sole_camber_area_m2
        message = str(error)
        assert "45" in message
        assert "camber_fit" in message, "the refusal must name the way out"

    def test_nearest_is_the_explicit_opt_in(self) -> None:
        result = loft_wedge(
            _outside_band_wedge(), camber_fit=CamberFit.NEAREST, **COARSE
        )
        assert isinstance(result, LoftedWedge)
        assert result.camber_fit is CamberFit.NEAREST

    def test_a_constructible_declaration_lofts_under_either_policy(self) -> None:
        wedge = build_reference_wedge()
        strict = loft_wedge(wedge, **COARSE)
        nearest = loft_wedge(wedge, camber_fit=CamberFit.NEAREST, **COARSE)
        assert strict.effective_camber_area_m2 == nearest.effective_camber_area_m2
        assert not strict.aggregate_camber_was_clamped

    def test_an_unknown_policy_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="camber_fit"):
            loft_wedge(build_reference_wedge(), camber_fit="nearest", **COARSE)  # type: ignore[arg-type]


class TestTheCallerCanDetectTheSubstitution:
    """Deliverable 1: the whole point -- public API only, no private imports."""

    def test_a_clamped_design_is_detectable_through_the_public_api(self) -> None:
        wedge = _outside_band_wedge()
        declared = wedge.sole_camber_area_m2

        result = loft_wedge(wedge, camber_fit=CamberFit.NEAREST, **COARSE)

        # This is the assertion the issue is about: a caller that declared
        # 45 mm^2 can find out, from the object it was handed, that the mesh
        # it received carries something else.
        assert result.aggregate_camber_was_clamped is True
        assert result.any_camber_was_clamped is True
        assert result.declared_camber_area_m2 == declared
        assert result.effective_camber_area_m2 != declared
        low, high = result.constructible_camber_range_m2
        assert low <= result.effective_camber_area_m2 <= high
        assert result.aggregate_camber_substitution_m2 == pytest.approx(
            result.effective_camber_area_m2 - declared
        )

    def test_an_unclamped_design_reports_no_substitution(self) -> None:
        """The flag must not be trivially true: a clean design says so.

        The unrelieved centre station is the one that carries the *declared*
        camber; relieved stations are a separate account, checked below.
        """
        wedge = build_reference_wedge()
        result = loft_wedge(wedge, camber_fit=CamberFit.NEAREST, **COARSE)
        assert result.aggregate_camber_was_clamped is False
        assert result.effective_camber_area_m2 == wedge.sole_camber_area_m2
        assert result.aggregate_camber_substitution_m2 == 0.0
        centre = next(
            s for s in result.stations if s.sole_width_m == wedge.sole_width_m
        )
        assert centre.was_clamped is False
        assert centre.camber_substitution_m2 == 0.0

    def test_relieved_stations_are_reported_station_by_station(self) -> None:
        """Relief scales the request; a narrow station may still not carry it.

        That substitution is derived rather than declared, so it never raises
        -- but it is recorded, which is what stops it being silent.
        """
        wedge = build_reference_wedge(
            heel_relief_fraction=0.30, toe_relief_fraction=0.05
        )
        result = loft_wedge(wedge, n_profile_points=24, n_stations=7)
        assert len(result.stations) == 7
        assert all(isinstance(s, StationCamber) for s in result.stations)
        heel = result.stations[0]
        assert heel.sole_width_m < wedge.sole_width_m
        assert heel.was_clamped is True
        assert heel.effective_camber_area_m2 != heel.requested_camber_area_m2
        assert result.clamped_stations
        assert set(result.clamped_stations) <= set(result.stations)

    def test_the_effective_value_survives_a_study_style_record(self) -> None:
        """What a sweep would actually log, in the demo harness's columns."""
        result = loft_wedge(
            _outside_band_wedge(), camber_fit=CamberFit.NEAREST, **COARSE
        )
        low, high = result.constructible_camber_range_m2
        record = {
            "declared_camber_mm2": result.declared_camber_area_m2 * 1e6,
            "effective_camber_mm2": result.effective_camber_area_m2 * 1e6,
            "aggregate_camber_was_clamped": result.aggregate_camber_was_clamped,
            "any_camber_was_clamped": result.any_camber_was_clamped,
            "clamped_station_count": len(result.clamped_stations),
            "constructible_camber_low_mm2": low * 1e6,
            "constructible_camber_high_mm2": high * 1e6,
        }
        assert record["aggregate_camber_was_clamped"] is True
        assert record["declared_camber_mm2"] == pytest.approx(45.0)
        assert record["effective_camber_mm2"] != pytest.approx(45.0)


class TestTheMeshIsUnchanged:
    """The clamp is correct and stays; only its observability changed."""

    def test_loft_wedge_carries_the_mesh_build_wedge_mesh_returns(self) -> None:
        wedge = build_reference_wedge()
        result = loft_wedge(wedge, **COARSE)
        mesh = build_wedge_mesh(wedge, **COARSE)
        assert result.mesh.vertices.shape == mesh.vertices.shape
        assert result.mesh.faces.shape == mesh.faces.shape
        assert check_mesh_validity(result.mesh).is_watertight_solid

    def test_the_clamp_still_fits_the_nearest_constructible_area(self) -> None:
        wedge = _outside_band_wedge()
        low, _ = constructible_camber_range_m2(wedge, n_points=24)
        result = loft_wedge(wedge, camber_fit=CamberFit.NEAREST, **COARSE)
        assert result.effective_camber_area_m2 == pytest.approx(low)


class TestDemoSweepRegression:
    """Pinned from the 77-point demo sweep that produced the issue's numbers.

    Base design ``sm9_54_f`` (48.0 mm^2 declared) swept over geometric bounce
    and sole width at the lofter's default 40 profile points.
    """

    @pytest.mark.parametrize(
        ("bounce_deg", "sole_width_m", "low_mm2", "high_mm2", "effective_mm2"),
        [
            (14.0, 0.016, 24.3911, 24.4934, 24.4934),
            (20.0, 0.020, 41.2998, 50.9807, 48.0),
            (26.0, 0.024, 61.5964, 89.3669, 61.5964),
        ],
    )
    def test_the_band_and_the_effective_value_match_the_sweep(
        self,
        bounce_deg: float,
        sole_width_m: float,
        low_mm2: float,
        high_mm2: float,
        effective_mm2: float,
    ) -> None:
        geometry = dataclasses.replace(
            get_preset("sm9_54_f").geometry,
            geometric_bounce=GeometricBounce(bounce_deg),
            sole_width_m=sole_width_m,
        )
        low, high = constructible_camber_range_m2(geometry, n_points=40)
        assert low * 1e6 == pytest.approx(low_mm2, abs=1e-3)
        assert high * 1e6 == pytest.approx(high_mm2, abs=1e-3)
        declared = geometry.sole_camber_area_m2
        effective = min(max(declared, low), high)
        assert effective * 1e6 == pytest.approx(effective_mm2, abs=1e-3)


class TestPresetsDeclareWhatTheyBuild:
    """A shipped grind must not declare a camber its own sole cannot carry."""

    @pytest.mark.parametrize("name", preset_names())
    def test_declared_camber_is_inside_its_own_band(self, name: str) -> None:
        geometry = get_preset(name).geometry
        for n_points in (24, 40):
            low, high = constructible_camber_range_m2(geometry, n_points=n_points)
            assert low <= geometry.sole_camber_area_m2 <= high, (
                f"{name} declares {geometry.sole_camber_area_m2 * 1e6:.3f} mm^2 "
                f"but admits {low * 1e6:.3f}-{high * 1e6:.3f} mm^2 at "
                f"n_points={n_points}"
            )

    @pytest.mark.parametrize("name", preset_names())
    def test_every_preset_lofts_under_the_strict_default(self, name: str) -> None:
        result = loft_wedge(get_preset(name).geometry, **COARSE)
        assert result.aggregate_camber_was_clamped is False


class TestTheFlagCannotUnderstateSubstitution:
    """The scope regression: a `False` boolean over refitted stations.

    ``camber_was_clamped`` compared the declared area against the band the
    *declared* sole width admits, so it answered only the aggregate question.
    Heel and toe relief narrows the sole toward the ends and a narrower sole
    admits a narrower band, so the relieved stations get refitted while the
    declaration itself is honoured -- and the flag read ``False`` beside a
    non-empty ``clamped_stations``.

    That is issue #8698's own failure mode (a substitution invisible to its
    caller) reappearing through the *simpler* check, which is the one most
    callers reach for.  These tests pin the invariant that closes it.
    """

    def test_the_broad_flag_cannot_be_false_while_stations_were_clamped(
        self,
    ) -> None:
        """The invariant, stated directly. This is the regression."""
        wedge = build_reference_wedge(
            heel_relief_fraction=0.30, toe_relief_fraction=0.05
        )
        result = loft_wedge(wedge, n_profile_points=24, n_stations=7)

        assert result.clamped_stations, "fixture must actually clamp a station"
        assert result.any_camber_was_clamped is True
        assert not (result.clamped_stations and not result.any_camber_was_clamped)

    def test_the_relieved_case_is_exactly_where_the_two_flags_diverge(self) -> None:
        """Aggregate honoured, stations refitted -- the misreported case."""
        wedge = build_reference_wedge(
            heel_relief_fraction=0.30, toe_relief_fraction=0.05
        )
        result = loft_wedge(wedge, n_profile_points=24, n_stations=7)

        # The declaration itself was constructible ...
        assert result.aggregate_camber_was_clamped is False
        assert result.effective_camber_area_m2 == wedge.sole_camber_area_m2
        # ... yet the sole that was built is not the sole that was asked for.
        assert result.clamped_stations
        assert result.any_camber_was_clamped is True

    @pytest.mark.parametrize("name", preset_names())
    def test_the_invariant_holds_on_every_shipped_preset(self, name: str) -> None:
        """Three of the six shipped grinds refit stations at full resolution."""
        result = loft_wedge(get_preset(name).geometry, camber_fit=CamberFit.NEAREST)
        if result.clamped_stations:
            assert result.any_camber_was_clamped is True
        assert result.any_camber_was_clamped >= result.aggregate_camber_was_clamped

    def test_the_shipped_default_preset_is_the_documented_case(self) -> None:
        """`sm9_58_m` is the preset the PR discussion measured: in band, refitted.

        Pinned by name and by number because the docstrings in ``lofting`` and
        ``report`` both cite it as the worked example.
        """
        result = loft_wedge(
            get_preset("sm9_58_m").geometry, camber_fit=CamberFit.NEAREST
        )
        declared_mm2 = result.declared_camber_area_m2 * 1e6
        low, high = result.constructible_camber_range_m2

        assert declared_mm2 == pytest.approx(42.0, abs=0.01)
        assert low * 1e6 < declared_mm2 < high * 1e6
        assert result.aggregate_camber_was_clamped is False
        assert len(result.clamped_stations) == 3
        assert result.any_camber_was_clamped is True

    def test_the_narrow_flag_keeps_its_narrow_meaning(self) -> None:
        """The rename must not have widened the aggregate flag by accident."""
        result = loft_wedge(
            _outside_band_wedge(), camber_fit=CamberFit.NEAREST, **COARSE
        )
        assert result.aggregate_camber_was_clamped is True
        assert result.aggregate_camber_substitution_m2 != 0.0

    def test_the_misleading_name_is_gone(self) -> None:
        """The old name must not survive as an alias reading the old value.

        Leaving it in place -- even deprecated -- would leave the misleading
        answer reachable, so the rename is a hard break by design.
        """
        result = loft_wedge(build_reference_wedge(), **COARSE)
        assert not hasattr(result, "camber_was_clamped")
        assert not hasattr(result, "camber_substitution_m2")
