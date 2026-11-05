"""An injected ShotOutcomeModel must be sampled with common random numbers (#14677)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LieRaster
from src.shared.python.sg_optimizer.mdp.action import ActionSet, ShotAction
from src.shared.python.sg_optimizer.mdp.value_iteration import HoleMDP
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.baseline_model import (
    BaselineShotModel,
)
from src.shared.python.sg_optimizer.shot_model.common_random import (
    CommonRandomModel,
)
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[4] / "data/sg_optimizer/baselines/pga_tour.yaml"
CONDITIONS = CourseConditions.benign()
PROFILE = PlayerProfile(name="p", baseline=str(YAML))
SHOT = ShotAction("7_iron", 0.0)


def _raster(cells: int = 14, res: float = 15.0) -> LieRaster:
    codes = np.full((cells, cells), LIE_CODES["fairway"], dtype=np.int8)
    codes[:, :2] = LIE_CODES["water"]
    mid = cells // 2
    codes[mid - 1 : mid + 1, mid - 1 : mid + 1] = LIE_CODES["holed"]
    half = cells // 2 * res
    return LieRaster(codes, res, (-half, -half), (0.0, 0.0), (-80.0, 0.0))


def _mdp(model: object | None, n: int = 16, seed: int = 0) -> HoleMDP:
    return HoleMDP(
        _raster(),
        PROFILE,
        load_baseline(YAML),
        CONDITIONS,
        ActionSet(clubs=("7_iron", "pw"), aim_grid_deg=np.array([-20.0, 0.0, 20.0])),
        n_samples=n,
        seed=seed,
        model=model,  # type: ignore[arg-type]
    )


def _crn(seed: int = 0) -> CommonRandomModel:
    return CommonRandomModel(BaselineShotModel(load_baseline(YAML)), seed)


def _draw(model: CommonRandomModel, shot: ShotAction = SHOT, n: int = 16):
    lie = LIE_CODES["fairway"]
    return model.sample(PROFILE, shot, lie, CONDITIONS, np.random.default_rng(), n)


def test_repeated_draws_are_identical_whatever_rng_is_passed() -> None:
    model = _crn()
    a, b = _draw(model), _draw(model)
    np.testing.assert_array_equal(a.along, b.along)
    np.testing.assert_array_equal(a.lateral, b.lateral)


def test_seed_decides_the_draw_and_actions_get_their_own() -> None:
    a = _draw(_crn(1))
    np.testing.assert_array_equal(a.along, _draw(_crn(1)).along)
    assert not np.array_equal(a.along, _draw(_crn(2)).along)
    assert not np.array_equal(a.along, _draw(_crn(1), ShotAction("pw", 0.0)).along)


def test_changing_the_player_after_first_use_is_refused() -> None:
    model = _crn()
    _draw(model)
    other = PlayerProfile(name="other", baseline=str(YAML))
    with pytest.raises(ValueError, match="fixed"):
        model.sample(
            other, SHOT, LIE_CODES["fairway"], CONDITIONS, np.random.default_rng(), 16
        )


def test_injected_model_backup_is_a_fixed_operator() -> None:
    mdp = _mdp(BaselineShotModel(load_baseline(YAML)))
    v0 = mdp._initial_value()
    first, _ = mdp.bellman_backup(v0)
    second, _ = mdp.bellman_backup(v0)
    np.testing.assert_array_equal(first, second)


def test_injected_model_solve_reaches_tolerance() -> None:
    mdp = _mdp(BaselineShotModel(load_baseline(YAML)))
    # Budget: water cells climb 1 per sweep to VALUE_CAP (30), so allow 120 sweeps.
    result = mdp.solve(tol=1e-3, max_iter=120)
    assert result.delta < 1e-3 and result.iterations < 120


def test_scalar_and_vectorized_backups_agree_tighter_than_the_old_bound() -> None:
    """Same grid and Gaussian: mean relative gap well under the old 0.5 bound."""
    vec = _mdp(None, n=200)
    sca = _mdp(BaselineShotModel(load_baseline(YAML)), n=200)
    v0 = vec._initial_value()
    a, _ = vec.bellman_backup(v0)
    b, _ = sca.bellman_backup(v0)
    valid = vec.raster.codes != LIE_CODES["holed"]
    rel = np.abs(a - b)[valid].mean() / max(b[valid].mean(), 1e-3)
    assert rel < 0.1, f"scalar vs vectorized relative gap {rel:.3f}"
