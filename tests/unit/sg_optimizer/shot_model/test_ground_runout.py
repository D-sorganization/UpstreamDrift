"""SHOW-B3b (#14610): bounce/roll rest position on the B6 slope bins."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model import ground_runout as gr
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    _impacts,
    standard_normals,
    TableGrid,
    build_grid_table,
)
from src.shared.python.sg_optimizer.shot_model.ground_runout import (
    GROUND_MODEL_ID,
    LANDING_COLUMNS,
    SlopeAwareGround,
)
from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
    PipelineLandingSimulator,
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


def _rest(
    slope: str, lie: str = "fairway", skill: str = "mid_iron", **kw: float
) -> np.ndarray:
    return GROUND(_landing(**kw), lie, slope, skill)


def test_there_is_one_slope_vocabulary_and_it_is_the_b6_one() -> None:
    assert gr.SLOPE_BINS is SLOPE_BINS  # the same object: no second table to drift
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
    rest = GROUND(land, "fairway", "flat", "mid_iron")
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
        GROUND(
            np.array([[100.0, 0.0]]), "fairway", "flat", "mid_iron"
        )  # no velocity columns
    with pytest.raises(ContractViolationError):
        GROUND(_landing(), "swamp", "flat", "mid_iron")
    with pytest.raises(ContractViolationError):
        GROUND(_landing(), "fairway", "uphill", "mid_iron")
    with pytest.raises(ContractViolationError):
        GROUND(_landing(), "fairway", "flat", "putter")


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
    run = {k: cell(rest, k).mean[0] - cell(carry, k).mean[0] for k in grid.slopes}
    assert run["flat"] > 0.0
    assert (
        run["up_5"] < run["flat"] < run["down_5"]
    )  # the slope bins alone move the runout
    assert rest.provenance["ground_model"] == GROUND_MODEL_ID
    assert "ground_model" not in carry.provenance  # carry-only hashes are unchanged


def test_ground_without_landing_state_columns_is_rejected() -> None:
    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",))
    with pytest.raises(ContractViolationError):
        build_grid_table(
            family, lambda i, s, w: _stub_sim(i, s, w)[:, :2], grid, n=8, ground=GROUND
        )


def _prov_table(ground):
    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",))
    return build_grid_table(family, _stub_sim, grid, n=16, seed=1, ground=ground)


def test_changing_a_coefficient_changes_the_table_hash(monkeypatch) -> None:
    base = _prov_table(GROUND)
    rec = gr.GROUND_PARAMETERS[
        "roll.sand"
    ]  # unused on a fairway grid: cells are unchanged
    monkeypatch.setitem(
        gr.GROUND_PARAMETERS, "roll.sand", replace(rec, value=rec.value + 0.1)
    )
    changed = _prov_table(GROUND)
    assert changed.cells == base.cells  # same cells, different physics: hash must move
    assert changed.table_hash != base.table_hash
    prov = base.provenance
    assert prov["ground_parameters"]["roll.sand"] == rec.value
    assert "stance" in prov["ground_slope_source"]
    # one lie bin is the impact lie AND the landing surface: declared and hashed
    assert "landing surface" in prov["ground_lie_source"]


def test_roll_fraction_of_carry_falls_from_driver_to_wedge_on_the_real_pipeline() -> (
    None
):
    family, sim = get_skill_family("tour"), PipelineLandingSimulator(landing_state=True)
    z = standard_normals(0, 12)
    frac = []
    for skill in (
        "driver",
        "fairway_wood",
        "long_iron",
        "mid_iron",
        "short_iron",
        "wedge",
    ):
        land = sim(_impacts(family.skills[skill].distribution, z), skill, (0.0, 0.0))
        rest = GROUND(land, "fairway", "flat", skill)
        frac.append(float((rest[:, 0] - land[:, 0]).mean() / land[:, 0].mean()))
    assert all(a >= b for a, b in zip(frac, frac[1:], strict=False)), frac


def test_green_side_skills_keep_carry_and_the_list_is_hashed() -> None:
    for skill in ("pitch", "chip", "bunker"):
        rest = GROUND(_landing(), "fairway", "flat", skill)
        np.testing.assert_array_equal(rest, _landing()[:, :2])
    assert set(GROUND.provenance["ground_carry_only_skills"]) == {
        "pitch",
        "chip",
        "bunker",
    }
    assert _prov_table(GROUND).provenance["ground_carry_only_skills"]


def test_ground_provenance_cannot_override_table_keys() -> None:
    class _Clash:
        provenance = {"seed": 99}

        def __call__(self, landing, lie, slope, skill):
            return GROUND(landing, lie, slope, skill)

    family = get_skill_family("tour")
    grid = TableGrid(("mid_iron",), ("fairway",), ("calm",))
    with pytest.raises(ContractViolationError):
        build_grid_table(family, _stub_sim, grid, n=8, ground=_Clash())
