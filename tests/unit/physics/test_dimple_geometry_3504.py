"""Unit tests for the parametric dimple-geometry drag model (#3504)."""

from __future__ import annotations

import pytest
from src.shared.python.physics.dimple_geometry import (
    MAX_DRAG_REDUCTION,
    RE_TRANSITION,
    DimpleGeometry,
    dimple_adjusted_cd,
)

# Reference Reynolds numbers
RE_FLIGHT = 1.5e5  # representative tour-ball flight
RE_LOW = 1.0e4  # well below dimple transition
BASE_CD = 0.45


def _optimal() -> DimpleGeometry:
    """Return a fully optimal dimple geometry (efficiency == 1)."""
    return DimpleGeometry(
        count=500,
        depth_mm=0.20,
        diameter_mm=3.5,
        coverage_fraction=0.85,
    )


def _tour_typical() -> DimpleGeometry:
    """A representative tour-style geometry (not maximal efficiency)."""
    return DimpleGeometry(
        count=380,
        depth_mm=0.20,
        diameter_mm=3.5,
        coverage_fraction=0.75,
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestDimpleGeometryConstruction:
    @pytest.mark.unit
    def test_optimal_geometry_efficiency_is_one(self) -> None:
        geom = _optimal()
        assert geom.efficiency == pytest.approx(1.0)

    @pytest.mark.unit
    def test_smooth_low_count_efficiency_is_zero(self) -> None:
        # Below all "good" thresholds.
        geom = DimpleGeometry(
            count=100,
            depth_mm=0.05,
            diameter_mm=2.5,
            coverage_fraction=0.40,
        )
        assert geom.efficiency == pytest.approx(0.0)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "count",
        [-1, 0, 1001, 5000],
    )
    def test_invalid_count_raises(self, count: int) -> None:
        with pytest.raises(ValueError):
            DimpleGeometry(
                count=count,
                depth_mm=0.20,
                diameter_mm=3.5,
                coverage_fraction=0.75,
            )

    @pytest.mark.unit
    def test_invalid_depth_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            DimpleGeometry(
                count=380,
                depth_mm=-0.01,
                diameter_mm=3.5,
                coverage_fraction=0.75,
            )

    @pytest.mark.unit
    def test_invalid_depth_too_large_raises(self) -> None:
        with pytest.raises(ValueError):
            DimpleGeometry(
                count=380,
                depth_mm=2.0,
                diameter_mm=3.5,
                coverage_fraction=0.75,
            )

    @pytest.mark.unit
    def test_invalid_coverage_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            DimpleGeometry(
                count=380,
                depth_mm=0.20,
                diameter_mm=3.5,
                coverage_fraction=1.5,
            )

    @pytest.mark.unit
    def test_invalid_coverage_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            DimpleGeometry(
                count=380,
                depth_mm=0.20,
                diameter_mm=3.5,
                coverage_fraction=-0.1,
            )

    @pytest.mark.unit
    def test_non_int_count_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            DimpleGeometry(
                count=380.0,  # type: ignore[arg-type]
                depth_mm=0.20,
                diameter_mm=3.5,
                coverage_fraction=0.75,
            )


# ---------------------------------------------------------------------------
# Cd model
# ---------------------------------------------------------------------------


class TestDimpleAdjustedCd:
    @pytest.mark.unit
    def test_optimal_at_flight_re_halves_smooth_cd(self) -> None:
        cd = dimple_adjusted_cd(_optimal(), RE_FLIGHT, BASE_CD)
        assert cd == pytest.approx(BASE_CD * (1.0 - MAX_DRAG_REDUCTION))

    @pytest.mark.unit
    def test_optimal_cd_in_tour_ball_band(self) -> None:
        # Industry data: tour balls at Re=1.5e5 have Cd ~ 0.22-0.28.
        # Fully optimal geometry sits at the lower bound of the band.
        cd = dimple_adjusted_cd(_optimal(), RE_FLIGHT, BASE_CD)
        assert 0.22 <= cd <= 0.28

    @pytest.mark.unit
    def test_tour_typical_geometry_reduces_drag(self) -> None:
        cd = dimple_adjusted_cd(_tour_typical(), RE_FLIGHT, BASE_CD)
        # A typical tour ball geometry must achieve a meaningful drag
        # reduction at flight Reynolds number.
        assert cd < BASE_CD * 0.85
        assert cd > BASE_CD * (1.0 - MAX_DRAG_REDUCTION) - 1e-9

    @pytest.mark.unit
    def test_suboptimal_geometry_reduces_less(self) -> None:
        sub = DimpleGeometry(
            count=300,
            depth_mm=0.15,
            diameter_mm=3.5,
            coverage_fraction=0.68,
        )
        cd_sub = dimple_adjusted_cd(sub, RE_FLIGHT, BASE_CD)
        cd_opt = dimple_adjusted_cd(_optimal(), RE_FLIGHT, BASE_CD)
        assert cd_sub > cd_opt
        assert cd_sub < BASE_CD  # still some reduction

    @pytest.mark.unit
    def test_low_re_returns_base_cd(self) -> None:
        cd = dimple_adjusted_cd(_optimal(), RE_LOW, BASE_CD)
        assert cd == pytest.approx(BASE_CD)

    @pytest.mark.unit
    def test_exactly_at_transition_re_applies_dimple_effect(self) -> None:
        cd_at = dimple_adjusted_cd(_optimal(), RE_TRANSITION, BASE_CD)
        # At the transition we activate the reduction.
        assert cd_at < BASE_CD

    @pytest.mark.unit
    def test_just_below_transition_returns_base(self) -> None:
        cd = dimple_adjusted_cd(_optimal(), RE_TRANSITION * 0.999, BASE_CD)
        assert cd == pytest.approx(BASE_CD)

    @pytest.mark.unit
    def test_monotonic_in_coverage(self) -> None:
        """More-optimal coverage should yield lower Cd at flight Re."""
        coverages = [0.66, 0.70, 0.75, 0.80, 0.85]
        cds = [
            dimple_adjusted_cd(
                DimpleGeometry(
                    count=380,
                    depth_mm=0.20,
                    diameter_mm=3.5,
                    coverage_fraction=c,
                ),
                RE_FLIGHT,
                BASE_CD,
            )
            for c in coverages
        ]
        # Strictly non-increasing.
        for prev, nxt in zip(cds, cds[1:], strict=False):
            assert nxt <= prev
        # And the extremes differ.
        assert cds[-1] < cds[0]

    @pytest.mark.unit
    def test_monotonic_in_count(self) -> None:
        counts = [280, 320, 380, 440, 500]
        cds = [
            dimple_adjusted_cd(
                DimpleGeometry(
                    count=c,
                    depth_mm=0.20,
                    diameter_mm=3.5,
                    coverage_fraction=0.75,
                ),
                RE_FLIGHT,
                BASE_CD,
            )
            for c in counts
        ]
        for prev, nxt in zip(cds, cds[1:], strict=False):
            assert nxt <= prev
        assert cds[-1] < cds[0]

    @pytest.mark.unit
    def test_non_positive_re_raises(self) -> None:
        with pytest.raises(ValueError):
            dimple_adjusted_cd(_optimal(), 0.0, BASE_CD)
        with pytest.raises(ValueError):
            dimple_adjusted_cd(_optimal(), -1.0, BASE_CD)

    @pytest.mark.unit
    def test_non_positive_base_cd_raises(self) -> None:
        with pytest.raises(ValueError):
            dimple_adjusted_cd(_optimal(), RE_FLIGHT, 0.0)
        with pytest.raises(ValueError):
            dimple_adjusted_cd(_optimal(), RE_FLIGHT, -0.5)

    @pytest.mark.unit
    def test_geometry_type_validated(self) -> None:
        with pytest.raises(TypeError):
            dimple_adjusted_cd(
                {"count": 380},  # type: ignore[arg-type]
                RE_FLIGHT,
                BASE_CD,
            )

    @pytest.mark.unit
    def test_reynolds_type_validated(self) -> None:
        with pytest.raises(TypeError):
            dimple_adjusted_cd(
                _optimal(),
                "1.5e5",  # type: ignore[arg-type]
                BASE_CD,
            )
