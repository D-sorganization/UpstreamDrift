"""SHOW-B6 (#14570): slope/stance bins and wind bins in the dispersion tables."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    DEFAULT_WINDS,
    WIND_BINS,
    TableGrid,
    build_grid_table,
    build_table,
)
from src.shared.python.sg_optimizer.shot_model.skill_family import IMPACT_FIELDS
from src.shared.python.sg_optimizer.shot_model.skill_registry import get_skill_family
from src.shared.python.sg_optimizer.shot_model.slope_wind_bins import (
    SLOPE_BINS,
    SLOPE_PARAMETERS,
    make_slope_modifier,
    wind_to_aim_frame,
)

pytestmark = pytest.mark.unit
COL = {name: i for i, name in enumerate(IMPACT_FIELDS)}
ROWS = np.tile(np.array([100.0, 1.0, 2.0, -3.0, 4.0, 5.0]), (50, 1))


def _mean(name: str, slope: str) -> float:
    return float(make_slope_modifier(slope)(ROWS)[:, COL[name]].mean())


def test_flat_is_the_exact_identity_and_input_is_untouched() -> None:
    out = make_slope_modifier("flat")(ROWS)
    assert np.array_equal(out, ROWS)
    make_slope_modifier("up_5")(ROWS)
    assert np.array_equal(ROWS[0], [100.0, 1.0, 2.0, -3.0, 4.0, 5.0])


def test_uphill_raises_and_downhill_lowers_the_attack_angle_and_both_lose_speed() -> (
    None
):
    assert (
        _mean("attack_angle_deg", "up_5") > -3.0 > _mean("attack_angle_deg", "down_5")
    )
    assert _mean("clubhead_speed_mph", "up_5") < 100.0
    assert _mean("clubhead_speed_mph", "down_5") < 100.0


def test_ball_above_feet_closes_the_face_and_below_opens_it() -> None:
    assert _mean("face_angle_deg", "above_5") < 1.0 < _mean("face_angle_deg", "below_5")
    assert _mean("club_path_deg", "above_5") > 2.0 > _mean("club_path_deg", "below_5")


def test_every_slope_parameter_is_a_labelled_assumption() -> None:
    assert SLOPE_PARAMETERS
    assert all(p.basis == "assumption" and p.source for p in SLOPE_PARAMETERS.values())
    assert set(SLOPE_BINS) == {"flat", "up_5", "down_5", "above_5", "below_5"}


def test_unknown_slope_and_bad_rows_are_rejected() -> None:
    with pytest.raises(ContractViolationError):
        make_slope_modifier("sideways")
    with pytest.raises(ContractViolationError):
        make_slope_modifier("up_5")(ROWS[:, :3])


def test_wind_resolves_into_head_and_cross_components() -> None:
    assert wind_to_aim_frame(10.0, 0.0) == pytest.approx((10.0, 0.0))
    assert wind_to_aim_frame(10.0, 180.0) == pytest.approx((-10.0, 0.0), abs=1e-9)
    head, cross = wind_to_aim_frame(10.0, 90.0)
    assert head == pytest.approx(0.0, abs=1e-9) and cross == pytest.approx(10.0)
    with pytest.raises(ContractViolationError):
        wind_to_aim_frame(-1.0, 0.0)


def test_default_winds_keep_the_original_bins_and_new_bins_exist() -> None:
    assert DEFAULT_WINDS == ("calm", "head_5", "tail_5", "cross_5")
    assert {"head_15", "tail_15", "cross_15"} <= set(WIND_BINS)


# Aim frame: lateral + is left, so a closed (smaller) face and a crosswind from
# the right both move the ball to + (matches the real-pipeline adapter below).
def _linear_sim(
    impacts: np.ndarray,
    skill: str,
    wind: tuple[float, float],
    slope: str,
    lie: str,
):
    _ = (slope, lie)
    return np.column_stack(
        [2.0 * impacts[:, 0] - wind[0], -3.0 * impacts[:, 1] + wind[1]]
    )


def test_slope_bins_without_a_modifier_are_rejected() -> None:
    with pytest.raises(ContractViolationError):
        build_grid_table(
            get_skill_family("tour"),
            _linear_sim,
            TableGrid(skills=("driver",), slopes=("up_5",)),
            n=50,
        )


def test_tables_carry_wind_and_slope_bins() -> None:
    family = get_skill_family("tour")
    table = build_grid_table(
        family,
        _linear_sim,
        TableGrid(
            skills=("driver",),
            winds=("calm", "tail_15", "cross_15"),
            slopes=("flat", "up_5", "above_5"),
        ),
        n=400,
        seed=1,
        slope_modifier=make_slope_modifier,
    )
    assert len(table.cells) == 1 * 3 * 3
    cell = {(c.wind, c.slope): c for c in table.cells}
    calm = cell[("calm", "flat")]
    assert cell[("tail_15", "flat")].mean[0] > calm.mean[0]  # downwind carries farther
    assert cell[("cross_15", "flat")].mean[1] == pytest.approx(calm.mean[1] + 15.0)
    assert cell[("calm", "above_5")].mean[1] > calm.mean[1]  # closed face draws left
    assert cell[("calm", "up_5")].mean[0] < calm.mean[0]  # speed lost on the slope
    assert calm == next(  # default build is unchanged for the flat/calm cell
        c
        for c in build_table(
            family, _linear_sim, skills=("driver",), n=400, seed=1
        ).cells
        if (c.wind, c.slope) == ("calm", "flat")
    )


def test_real_pipeline_wind_and_slope_cells_move_the_right_way() -> None:
    """The B3 real-pipeline adapter (not the linear stand-in) honours the new bins."""
    from src.shared.python.physics.rust_kernel import is_rust_available
    from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
        PipelineLandingSimulator,
    )

    if not is_rust_available():
        pytest.skip("upstream-physics Rust kernel required for flight pipeline")

    table = build_grid_table(
        get_skill_family("tour"),
        PipelineLandingSimulator(),
        TableGrid(
            skills=("driver",),
            winds=("calm", "head_15", "tail_15", "cross_15"),
            slopes=("flat", "up_5", "above_5"),
        ),
        n=40,
        seed=1,
        slope_modifier=make_slope_modifier,
    )
    cell = {(c.wind, c.slope): c for c in table.cells}
    calm = cell[("calm", "flat")]
    assert (
        cell[("tail_15", "flat")].mean[0]
        > calm.mean[0]
        > cell[("head_15", "flat")].mean[0]
    )
    assert abs(cell[("cross_15", "flat")].mean[1] - calm.mean[1]) > 10.0
    assert cell[("calm", "above_5")].mean[1] > calm.mean[1]  # closed face draws left
    assert cell[("calm", "up_5")].mean[0] != calm.mean[0]
