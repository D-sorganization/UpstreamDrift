"""Common random numbers make the Bellman backup a fixed operator (#14660)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LieRaster
from src.shared.python.sg_optimizer.mdp.action import ActionSet
from src.shared.python.sg_optimizer.mdp.value_iteration import HoleMDP
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[4] / "data/sg_optimizer/baselines/pga_tour.yaml"


def _mdp(seed: int = 0, aims: np.ndarray | None = None) -> HoleMDP:
    cells, res = 24, 10.0
    codes = np.full((cells, cells), LIE_CODES["fairway"], dtype=np.int8)
    codes[:, :4] = LIE_CODES["water"]
    codes[cells // 2 - 1 : cells // 2 + 2, cells // 2 - 1 : cells // 2 + 2] = LIE_CODES[
        "holed"
    ]
    half = cells // 2 * res
    raster = LieRaster(codes, res, (-half, -half), (0.0, 0.0), (-100.0, 0.0))
    return HoleMDP(
        raster,
        PlayerProfile(name="p", baseline=str(YAML)),
        load_baseline(YAML),
        CourseConditions.benign(),
        ActionSet(
            clubs=("7_iron", "pw"),
            aim_grid_deg=np.arange(-180.0, 180.0, 30.0) if aims is None else aims,
        ),
        n_samples=32,
        seed=seed,
    )


def test_same_value_function_gives_an_identical_backup() -> None:
    mdp = _mdp()
    v0 = mdp._initial_value()
    first, _ = mdp.bellman_backup(v0)
    second, _ = mdp.bellman_backup(v0)
    np.testing.assert_array_equal(first, second)


def test_equal_seeds_reproduce_and_different_seeds_differ() -> None:
    v0 = _mdp()._initial_value()
    a, _ = _mdp(1).bellman_backup(v0)
    b, _ = _mdp(1).bellman_backup(v0)
    c, _ = _mdp(2).bellman_backup(v0)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_solve_reaches_tolerance() -> None:
    result = _mdp().solve(tol=1e-3, max_iter=300)
    assert result.delta < 1e-3
    assert result.iterations < 300


def test_cells_with_no_way_to_the_hole_are_capped_and_the_solve_still_converges() -> (
    None
):
    """The cap is the safety bound for a hole the player cannot actually reach."""
    mdp = _mdp()
    codes = mdp.raster.codes.copy()
    codes[codes == LIE_CODES["fairway"]] = LIE_CODES["water"]  # only the hole is dry
    start = mdp.raster.world_to_index(-100.0, 0.0)
    codes[start] = LIE_CODES["fairway"]
    mdp = HoleMDP(
        LieRaster(
            codes,
            mdp.raster.resolution_yd,
            mdp.raster.origin,
            mdp.raster.pin,
            mdp.raster.tee,
        ),
        mdp.profile,
        mdp.baseline,
        mdp.conditions,
        mdp.actions,
        n_samples=mdp.n_samples,
        seed=0,
    )
    result = mdp.solve(tol=1e-3, max_iter=300)
    assert result.delta < 1e-3
    assert result.value[start] == mdp.VALUE_CAP
    assert float(result.value.max()) <= mdp.VALUE_CAP


def _demo_mdp() -> HoleMDP:
    """Fairway, water right of it, a holed green 380 yd out; default aim grid."""
    from src.shared.python.sg_optimizer.course.rasterize import (
        CircleFeature,
        RectFeature,
        SyntheticHole,
        rasterize_synthetic,
    )
    from src.shared.python.sg_optimizer.mdp.action import default_action_set

    hole = SyntheticHole(
        name="par4",
        par=4,
        tee=(0.0, 0.0),
        pin=(380.0, 0.0),
        bbox=(-20.0, 420.0, -60.0, 60.0),
        features=(
            RectFeature("fairway", 180.0, 330.0, -18.0, 18.0),
            RectFeature("water", 200.0, 330.0, -60.0, -26.0),
            CircleFeature("green", 380.0, 0.0, 14.0),
        ),
    )
    raster = rasterize_synthetic(hole, resolution_yd=20.0)
    codes = raster.codes.copy()
    codes[codes == LIE_CODES["green"]] = LIE_CODES["holed"]
    raster = LieRaster(
        codes, raster.resolution_yd, raster.origin, raster.pin, raster.tee
    )
    return HoleMDP(
        raster,
        PlayerProfile(name="p", baseline=str(YAML)),
        load_baseline(YAML),
        CourseConditions.benign(),
        default_action_set(),
        n_samples=24,
        seed=0,
    )


def test_demo_par4_with_water_converges_within_a_sweep_budget() -> None:
    """Fairway, water right, green: converges, and the tee is no worse than the cap."""
    mdp = _demo_mdp()
    result = mdp.solve(tol=1e-3, max_iter=80)
    assert result.delta < 1e-3 and result.iterations <= 80
    tee = mdp.raster.world_to_index(0.0, 0.0)
    assert 1.0 < result.value[tee] < mdp.VALUE_CAP


def test_tee_value_does_not_depend_on_the_cap() -> None:
    """The cap is a safety bound: moving it must not move the answer."""
    tee_values = []
    for cap in (30.0, 100.0):
        mdp = _demo_mdp()
        mdp.VALUE_CAP = cap
        result = mdp.solve(tol=1e-3, max_iter=200)
        assert result.delta < 1e-3
        tee_values.append(result.value[mdp.raster.world_to_index(0.0, 0.0)])
    assert tee_values[0] == pytest.approx(tee_values[1], abs=1e-2)


def test_no_playable_cell_sits_at_the_cap() -> None:
    """Overshoots can aim back at the pin, so no non-hazard cell is capped."""
    mdp = _demo_mdp()
    result = mdp.solve(tol=1e-3, max_iter=200)
    codes = mdp.raster.codes
    playable = ~np.isin(codes, [LIE_CODES["water"], LIE_CODES["ob"]])
    assert float(result.value[playable].max()) < mdp.VALUE_CAP - 1.0
