"""SHOW-B3b (#14610): bounce/roll rest position on the B6 slope bins."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model import ground_runout as gr
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    TableGrid,
    build_grid_table,
)
from src.shared.python.sg_optimizer.shot_model.ground_runout import (
    GROUND_MODEL_ID,
    LANDING_COLUMNS,
    SlopeAwareGround,
)
from src.shared.python.sg_optimizer.shot_model.skill_registry import get_skill_family
from src.shared.python.sg_optimizer.shot_model.slope_wind_bins import (
    SLOPE_BINS,
    make_slope_modifier,
)

pytestmark = pytest.mark.unit
GROUND = SlopeAwareGround()


def _landing(carry: float = 120.0, speed: float = 35.0, drop: float = 25.0):
    """One landing row: x, y (yd), vx, vy, vz (m/s); vz < 0 is descending."""
    return np.array([[carry, 2.0, speed, 0.0, -drop]])


def _rest(slope: str, lie: str = "fairway", **kw) -> np.ndarray:
    return GROUND(_landing(**kw), lie, slope)


def test_there_is_one_slope_vocabulary_and_it_is_the_b6_one() -> None:
    assert not hasattr(gr, "SLOPE_BINS")  # no second bin table to drift
    for slope in SLOPE_BINS:
        assert _rest(slope).shape == (1, 2)


def test_uphill_stops_shorter_and_downhill_runs_further_than_flat() -> None:
    flat, up, down = (_rest(s)[0, 0] for s in ("flat", "up_5", "down_5"))
    assert up < flat < down


def test_ball_above_or_below_feet_bins_do_not_change_runout() -> None:
    flat = _rest("flat")
    np.testing.assert_allclose(_rest("above_5"), flat)
    np.testing.assert_allclose(_rest("below_5"), flat)


def test_rest_is_never_behind_carry_and_roll_follows_the_landing_direction() -> None:
    land = np.array([[100.0, 0.0, 0.0, 30.0, -20.0]])  # landing sideways (+y)
    rest = GROUND(land, "fairway", "flat")
    assert rest[0, 0] == pytest.approx(100.0) and rest[0, 1] > 0.0
    assert _rest("up_5", speed=1.0)[0, 0] >= 120.0


def test_steeper_descent_and_rougher_lie_shorten_the_roll() -> None:
    assert _rest("flat", drop=45.0)[0, 0] < _rest("flat", drop=10.0)[0, 0]
    fairway, rough, sand = (
        _rest("flat", lie)[0, 0] for lie in ("fairway", "rough", "sand")
    )
    assert sand < rough < fairway


def test_missing_landing_inputs_and_unknown_names_fail_closed() -> None:
    with pytest.raises(ContractViolationError):
        GROUND(np.array([[100.0, 0.0]]), "fairway", "flat")  # no velocity columns
    with pytest.raises(ContractViolationError):
        GROUND(_landing(), "swamp", "flat")
    with pytest.raises(ContractViolationError):
        GROUND(_landing(), "fairway", "uphill")


def test_every_coefficient_is_a_labelled_assumption() -> None:
    assert GROUND.model_id == GROUND_MODEL_ID
    assert gr.GROUND_PARAMETERS
    assert all(
        p.basis == "assumption" and p.source for p in gr.GROUND_PARAMETERS.values()
    )
    assert LANDING_COLUMNS[:2] == ("x_yd", "y_yd")


def _stub_sim(impacts, skill, wind):
    """Landing state from the impact rows: speed drives carry and descent is fixed."""
    n = len(impacts)
    carry = impacts[:, 0] * 2.0
    return np.column_stack(
        [carry, np.zeros(n), np.full(n, 30.0), np.zeros(n), np.full(n, -20.0)]
    )


def test_table_cells_report_rest_not_carry_and_slopes_change_them() -> None:
    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",), ("flat", "up_5", "down_5"))
    kw = {"n": 64, "seed": 3, "slope_modifier": make_slope_modifier}
    carry = build_grid_table(
        family, lambda i, s, w: _stub_sim(i, s, w)[:, :2], grid, **kw
    )
    rest = build_grid_table(family, _stub_sim, grid, ground=GROUND, **kw)
    cell = lambda t, slope: t.lookup("mid_iron", "mid_iron", "fairway", "calm", slope)  # noqa: E731
    assert cell(rest, "flat").mean[0] > cell(carry, "flat").mean[0]
    assert (
        cell(rest, "up_5").mean[0]
        < cell(rest, "flat").mean[0]
        < cell(rest, "down_5").mean[0]
    )
    assert rest.provenance["ground_model"] == GROUND_MODEL_ID
    assert "ground_model" not in carry.provenance  # carry-only hashes are unchanged


def test_ground_without_landing_state_columns_is_rejected() -> None:
    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",))
    with pytest.raises(ContractViolationError):
        build_grid_table(
            family, lambda i, s, w: _stub_sim(i, s, w)[:, :2], grid, n=8, ground=GROUND
        )
