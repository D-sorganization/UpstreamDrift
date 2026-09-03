"""ADR-0046 G0 drift gate: dispersion, UD stack vs vendored Tools stack.

Both stacks export a function literally named ``analyze_dispersion`` and both
consume the same two columns of the same 160-shot session. **They do not
compute the same statistic.** The name is the only thing they share beyond the
sample count and the lateral mean.

    UD    ``launch_monitor/dispersion.py`` — 2-D target-relative summary:
          median centre, 95% covariance ellipse (major/minor/angle/area) and
          radial error RMSE/p50/p90 about that median centre. **No unit
          conversion at all**: results come back in whatever unit the frame
          carries.
    Tools ``rate_of_closure/launch_monitor_performance.py`` — 1-D lateral
          summary: mean/sd/RMS of the lateral column plus left/centre/right
          counts, **always converted to yards** and tagged ``unit="yd"``.

AGREE — asserted exactly
    * Sample count: 160 == 160.
    * Lateral mean, up to the declared unit conversion. UD 0.75830069375 m,
      Tools 0.82928772282370955 yd, ratio 1.0936132983377076 against
      YARDS_PER_METRE 1.0936132983377078 (2 ulp).
    * Carry mean, same relationship: UD ``mean_forward`` 139.79610284375 m vs
      the mean of Tools' ``DispersionPoint.carry_yards`` 152.88287712571088 yd.

DIFFER — documented and pinned below
    D6. Disjoint result surfaces. The two dataclasses share **zero** field
        names. UD cannot report lateral sd/RMS or left/centre/right counts;
        Tools cannot report the ellipse, the median centre, or radial error.
    D7. "RMS" means different things. Tools' ``rms_yards`` 8.39694421985684 is
        1-D about zero lateral; UD's ``radial_rmse`` 11.364728588362174 is 2-D
        about the median centre. Same fixture, 35% apart, not a unit factor.
    D8. Sample floor. UD raises below 3 complete shots (it needs a covariance);
        Tools accepts 1.
    D9. Unit contract. Tools rejects any unit that is not ``yd``/``m``; UD
        never declares or converts a unit, so a metre-valued frame silently
        yields metre-valued "dispersion".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tools.launch_monitor_model.dispersion import (
    analyze_dispersion as ud_analyze_dispersion,
)
from tests.integration.launch_monitor_drift.conftest import (
    YARDS_PER_METRE,
    require_vendored_tools_stack,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

require_vendored_tools_stack()

from rate_of_closure.launch_monitor_performance import (  # noqa: E402
    DispersionRequest,
    analyze_dispersion as tools_analyze_dispersion,
)

FORWARD_COLUMN = "carry_distance_metres"
LATERAL_COLUMN = "lateral_carry_metres"
EXPECTED_SAMPLE_COUNT = 160

UD_MEAN_LATERAL_METRES = 0.75830069375000009
TOOLS_MEAN_LATERAL_YARDS = 0.82928772282370955
UD_MEAN_FORWARD_METRES = 139.79610284375
TOOLS_MEAN_CARRY_YARDS = 152.88287712571088

# D6/D7 pins: each stack's exclusive numbers on this fixture.
UD_ONLY_CENTER_FORWARD = 140.46910150000002
UD_ONLY_CENTER_LATERAL = 0.8098719999999999
UD_ONLY_ELLIPSE_MAJOR = 41.200179743913914
UD_ONLY_ELLIPSE_MINOR = 37.501804583285306
UD_ONLY_ELLIPSE_ANGLE_RAD = 3.0692190898528535
UD_ONLY_AREA_95 = 1213.5038500346534
UD_ONLY_RADIAL_RMSE = 11.364728588362174
UD_ONLY_RADIAL_P50 = 9.728486185536813
UD_ONLY_RADIAL_P90 = 16.386562069294477
TOOLS_ONLY_STANDARD_DEVIATION_YARDS = 8.382128584176664
TOOLS_ONLY_RMS_YARDS = 8.39694421985684
TOOLS_ONLY_LEFT_COUNT = 73
TOOLS_ONLY_CENTER_COUNT = 0
TOOLS_ONLY_RIGHT_COUNT = 87


def _tools_request() -> DispersionRequest:
    return DispersionRequest(
        lateral_column=LATERAL_COLUMN,
        carry_column=FORWARD_COLUMN,
        lateral_unit="m",
        carry_unit="m",
    )


@pytest.fixture(scope="module")
def ud_result(session_frame: pd.DataFrame):
    return ud_analyze_dispersion(
        session_frame, forward=FORWARD_COLUMN, lateral=LATERAL_COLUMN
    )


@pytest.fixture(scope="module")
def tools_result(session_frame: pd.DataFrame):
    return tools_analyze_dispersion(session_frame, _tools_request())


def test_sample_count_agrees(ud_result, tools_result) -> None:
    """AGREE: both stacks retain all 160 complete shots."""
    assert ud_result.sample_count == EXPECTED_SAMPLE_COUNT
    assert len(tools_result.points) == EXPECTED_SAMPLE_COUNT


def test_lateral_mean_agrees_up_to_the_pinned_unit_ratio(
    ud_result, tools_result
) -> None:
    """AGREE (ratio-pinned): the only difference is metres vs yards."""
    assert ud_result.mean_lateral == pytest.approx(UD_MEAN_LATERAL_METRES, rel=1e-12)
    assert tools_result.mean_lateral_yards == pytest.approx(
        TOOLS_MEAN_LATERAL_YARDS, rel=1e-12
    )
    ratio = tools_result.mean_lateral_yards / ud_result.mean_lateral
    assert ratio == pytest.approx(YARDS_PER_METRE, rel=1e-12), (
        f"lateral-mean ratio {ratio!r} is no longer the declared unit factor"
    )


def test_carry_mean_agrees_up_to_the_pinned_unit_ratio(ud_result, tools_result) -> None:
    """AGREE (ratio-pinned): Tools keeps carry only inside its point list."""
    tools_carry_mean = float(
        np.mean([point.carry_yards for point in tools_result.points])
    )
    assert ud_result.mean_forward == pytest.approx(UD_MEAN_FORWARD_METRES, rel=1e-12)
    assert tools_carry_mean == pytest.approx(TOOLS_MEAN_CARRY_YARDS, rel=1e-12)
    ratio = tools_carry_mean / ud_result.mean_forward
    assert ratio == pytest.approx(YARDS_PER_METRE, rel=1e-12)


def test_divergence_d6_result_surfaces_are_disjoint(ud_result, tools_result) -> None:
    """DIFFER (D6): zero shared field names between the two dispersions."""
    ud_fields = set(type(ud_result).__dataclass_fields__)
    tools_fields = set(type(tools_result).__dataclass_fields__)

    assert ud_fields == {
        "sample_count",
        "center_forward",
        "center_lateral",
        "mean_forward",
        "mean_lateral",
        "ellipse_major",
        "ellipse_minor",
        "ellipse_angle_rad",
        "area_95",
        "radial_rmse",
        "radial_p50",
        "radial_p90",
    }
    assert tools_fields == {
        "unit",
        "points",
        "mean_lateral_yards",
        "standard_deviation_yards",
        "rms_yards",
        "left_count",
        "center_count",
        "right_count",
        "formula",
    }
    assert ud_fields & tools_fields == set()


def test_divergence_d6_only_ud_computes_the_ellipse(ud_result) -> None:
    """DIFFER (D6): the covariance ellipse exists only in the UD stack."""
    assert ud_result.center_forward == pytest.approx(UD_ONLY_CENTER_FORWARD, rel=1e-12)
    assert ud_result.center_lateral == pytest.approx(UD_ONLY_CENTER_LATERAL, rel=1e-12)
    assert ud_result.ellipse_major == pytest.approx(UD_ONLY_ELLIPSE_MAJOR, rel=1e-12)
    assert ud_result.ellipse_minor == pytest.approx(UD_ONLY_ELLIPSE_MINOR, rel=1e-12)
    assert ud_result.ellipse_angle_rad == pytest.approx(
        UD_ONLY_ELLIPSE_ANGLE_RAD, rel=1e-12
    )
    assert ud_result.area_95 == pytest.approx(UD_ONLY_AREA_95, rel=1e-12)


def test_divergence_d6_only_tools_computes_lateral_spread_and_side_counts(
    session_frame: pd.DataFrame, tools_result
) -> None:
    """DIFFER (D6): lateral sd/RMS and side counts exist only in Tools.

    Both Tools numbers are exactly ``YARDS_PER_METRE`` times the statistic of
    the very column the UD stack consumes unconverted, so the gap is a missing
    UD *output*, not a different definition.
    """
    lateral_metres = session_frame[LATERAL_COLUMN].to_numpy(float)

    assert tools_result.standard_deviation_yards == pytest.approx(
        TOOLS_ONLY_STANDARD_DEVIATION_YARDS, rel=1e-12
    )
    assert tools_result.standard_deviation_yards == pytest.approx(
        float(np.std(lateral_metres, ddof=1)) * YARDS_PER_METRE, rel=1e-15
    )
    assert tools_result.rms_yards == pytest.approx(TOOLS_ONLY_RMS_YARDS, rel=1e-12)
    assert tools_result.rms_yards == pytest.approx(
        float(np.sqrt(np.mean(np.square(lateral_metres)))) * YARDS_PER_METRE,
        rel=1e-15,
    )
    assert (
        tools_result.left_count,
        tools_result.center_count,
        tools_result.right_count,
    ) == (TOOLS_ONLY_LEFT_COUNT, TOOLS_ONLY_CENTER_COUNT, TOOLS_ONLY_RIGHT_COUNT)
    assert tools_result.left_count + tools_result.right_count == EXPECTED_SAMPLE_COUNT


def test_divergence_d7_rms_is_a_different_estimand_in_each_stack(
    ud_result, tools_result
) -> None:
    """DIFFER (D7): 1-D-about-zero vs 2-D-about-the-median-centre.

    ``rms_yards`` 8.39694 yd and ``radial_rmse`` 11.36473 m are not related by
    the unit factor and must never be reconciled by renaming.
    """
    assert ud_result.radial_rmse == pytest.approx(UD_ONLY_RADIAL_RMSE, rel=1e-12)
    assert ud_result.radial_p50 == pytest.approx(UD_ONLY_RADIAL_P50, rel=1e-12)
    assert ud_result.radial_p90 == pytest.approx(UD_ONLY_RADIAL_P90, rel=1e-12)

    ratio = ud_result.radial_rmse / tools_result.rms_yards
    assert ratio == pytest.approx(1.3534362371357913, rel=1e-9)
    assert abs(ratio - YARDS_PER_METRE) > 0.25


def test_divergence_d8_minimum_sample_floors_differ(
    session_frame: pd.DataFrame,
) -> None:
    """DIFFER (D8): UD needs three complete shots; Tools accepts one."""
    two_shots = session_frame.head(2)

    with pytest.raises(ValueError, match="At least three complete shots"):
        ud_analyze_dispersion(two_shots, forward=FORWARD_COLUMN, lateral=LATERAL_COLUMN)

    tools = tools_analyze_dispersion(two_shots, _tools_request())
    assert len(tools.points) == 2

    single = tools_analyze_dispersion(session_frame.head(1), _tools_request())
    assert len(single.points) == 1
    assert single.standard_deviation_yards == 0.0


def test_divergence_d9_unit_contracts_differ(
    session_frame: pd.DataFrame, ud_result, tools_result
) -> None:
    """DIFFER (D9): Tools declares and validates units; UD does neither."""
    assert tools_result.unit == "yd"
    assert not hasattr(ud_result, "unit")

    with pytest.raises(ValueError, match="distance unit must be"):
        tools_analyze_dispersion(
            session_frame,
            DispersionRequest(
                lateral_column=LATERAL_COLUMN,
                carry_column=FORWARD_COLUMN,
                lateral_unit="ft",
                carry_unit="m",
            ),
        )

    # UD accepts the same frame with no unit declaration and returns metres.
    assert ud_result.mean_lateral == pytest.approx(UD_MEAN_LATERAL_METRES, rel=1e-12)
