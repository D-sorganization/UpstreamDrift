"""A design space must be checkable against the constructible camber band.

Issue #8698: a sensitivity run over sole width or bounce silently moves the
camber area whenever a corner of the box leaves the band a convex, monotone
sole admits.  ``MorrisDesign``, ``SaltelliDesign`` and ``SobolIndices`` would
then attribute variance to a factor the user believes is pinned, and nothing
in the artifact would say so.  The space can now be screened before the first
solver call.
"""

from __future__ import annotations

import pytest

from bunkershot3d.geometry.presets import get_preset
from bunkershot3d.study.analytic_benchmarks import ishigami_space
from bunkershot3d.study.design_space import DesignSpace

pytestmark = pytest.mark.unit

BASE = get_preset("sm9_54_f").geometry


class TestDesignSpaceScreensTheCamberBand:
    def test_the_demo_sweep_box_is_flagged(self) -> None:
        """The exact box that produced the issue's 40 clamped points."""
        space = DesignSpace.from_bounds(
            {
                "geometric_bounce_deg": (14.0, 26.0),
                "sole_width_mm": (16.0, 24.0),
            },
            units={"geometric_bounce_deg": "deg", "sole_width_mm": "mm"},
        )
        findings = space.check_wedge_camber(BASE)
        assert findings
        assert any("48" in finding for finding in findings)

    def test_a_box_that_stays_inside_the_band_is_clean(self) -> None:
        space = DesignSpace.from_bounds(
            {"sole_width_mm": (20.5, 21.5)}, units={"sole_width_mm": "mm"}
        )
        assert space.check_wedge_camber(BASE) == ()

    def test_a_space_with_no_sole_parameters_is_not_screened(self) -> None:
        assert ishigami_space().check_wedge_camber(BASE) == ()

    def test_a_camber_parameter_is_screened_against_its_own_bounds(self) -> None:
        space = DesignSpace.from_bounds(
            {"sole_camber_area_mm2": (30.0, 70.0)},
            units={"sole_camber_area_mm2": "mm2"},
        )
        findings = space.check_wedge_camber(BASE)
        assert findings
        assert any("sole_camber_area_mm2" in finding for finding in findings)

    def test_marketed_bounce_is_understood_too(self) -> None:
        space = DesignSpace.from_bounds(
            {"marketed_bounce_deg": (4.0, 20.0)},
            units={"marketed_bounce_deg": "deg"},
        )
        assert space.check_wedge_camber(BASE)

    def test_a_non_geometry_argument_is_a_type_error(self) -> None:
        space = DesignSpace.from_bounds({"sole_width_mm": (16.0, 24.0)})
        with pytest.raises(TypeError, match="WedgeGeometry"):
            space.check_wedge_camber("sm9_54_f")  # type: ignore[arg-type]

    def test_findings_name_the_parameter_and_quote_the_band(self) -> None:
        space = DesignSpace.from_bounds(
            {"sole_width_mm": (16.0, 24.0)}, units={"sole_width_mm": "mm"}
        )
        findings = space.check_wedge_camber(BASE)
        assert findings
        joined = " ".join(findings)
        assert "sole_width_mm" in joined
        assert "mm^2" in joined
