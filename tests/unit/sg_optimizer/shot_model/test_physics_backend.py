"""Physics backend on dispersion tables, shrinkage and strategy report (#14568)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LieRaster
from src.shared.python.sg_optimizer.mdp.action import ActionSet, ShotAction
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.dispersion_table import build_table
from src.shared.python.sg_optimizer.shot_model.player_profile import (
    ClubSkill,
    PlayerProfile,
)
from src.shared.python.sg_optimizer.shot_model.registry import (
    available_backends,
    get_shot_model,
)
from src.shared.python.sg_optimizer.shot_model.shrinkage import shrink_profile
from src.shared.python.sg_optimizer.shot_model.skill_fit import shrinkage_weight
from src.shared.python.sg_optimizer.shot_model.skill_registry import get_skill_family
from src.shared.python.sg_optimizer.shot_model.table_model import (
    CLUB_SKILL,
    TableShotModel,
    register_physics_backend,
)
from src.shared.python.sg_optimizer.strategy import (
    SolveSettings,
    StrategyProblem,
    compare_strategies,
)

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[4] / "data/sg_optimizer/baselines/pga_tour.yaml"
_FW = LIE_CODES["fairway"]


def _sim(impacts: np.ndarray, skill: str, wind: tuple[float, float]):
    """Stand-in physics: along = 2 * speed, lateral = 3 * face."""
    return np.stack([2.0 * impacts[:, 0], 3.0 * impacts[:, 1]], axis=1)


def _table():
    return build_table(
        get_skill_family("tour"), _sim, skills=("driver", "wedge"), n=2000, seed=1
    )


def _profile(**kw) -> PlayerProfile:
    return PlayerProfile(name="p", baseline=str(YAML), **kw)


def test_weight_limits_and_strength() -> None:
    assert shrinkage_weight(0, 10) == 0.0
    assert shrinkage_weight(10, 10) == pytest.approx(0.5)
    assert shrinkage_weight(10_000, 10) > 0.99
    assert shrinkage_weight(10, 100) < shrinkage_weight(10, 1)
    with pytest.raises(ContractViolationError):
        shrinkage_weight(-1, 10)


def test_shrink_profile_pulls_toward_baseline() -> None:
    raw = _profile(clubs={"7_iron": ClubSkill(skill_mult_lat=2.0, distance_offset=-10)})
    none = shrink_profile(raw, {}, strength=20).clubs["7_iron"]
    half = shrink_profile(raw, {"7_iron": 20}, strength=20).clubs["7_iron"]
    full = shrink_profile(raw, {"7_iron": 10**6}, strength=20).clubs["7_iron"]
    assert (none.skill_mult_lat, none.distance_offset) == (1.0, 0.0)
    assert half.skill_mult_lat == pytest.approx(1.5)
    assert half.distance_offset == pytest.approx(-5.0)
    assert full.skill_mult_lat == pytest.approx(2.0, abs=1e-3)


def test_club_skill_map_covers_baseline_clubs() -> None:
    assert set(load_baseline(YAML).clubs) <= set(CLUB_SKILL)


def test_table_model_samples_cell_moments_and_lie_scaling() -> None:
    table, bag = _table(), load_baseline(YAML)
    model = TableShotModel(table, get_shot_model("baseline", bag))
    cell = table.lookup("driver", "driver", "fairway", "calm", "flat")
    args = (_profile(), ShotAction("driver", 0.0))
    cond, rng = CourseConditions.tournament(), np.random.default_rng(2)
    fw = model.sample(*args, _FW, cond, rng, 20000)
    assert fw.along.mean() == pytest.approx(cell.mean[0], abs=1.0)
    assert fw.lateral.std() == pytest.approx(cell.sd[1], rel=0.05)
    ro = model.sample(*args, LIE_CODES["rough"], cond, np.random.default_rng(2), 20000)
    assert ro.along.mean() < fw.along.mean()


def test_table_model_falls_back_to_baseline_for_uncovered_club() -> None:
    bag = load_baseline(YAML)
    base = get_shot_model("baseline", bag)
    model = TableShotModel(_table(), base)  # table has no mid_iron cell
    args = (_profile(), ShotAction("7_iron", 0.0), _FW, CourseConditions.benign())
    a = model.sample(*args, np.random.default_rng(4), 50)
    b = base.sample(*args, np.random.default_rng(4), 50)
    np.testing.assert_array_equal(a.along, b.along)


def test_physics_backend_registers_by_name() -> None:
    from src.shared.python.sg_optimizer.shot_model import registry

    saved = dict(registry._BACKENDS)
    try:
        register_physics_backend(_table())
        assert "physics" in available_backends()
        assert isinstance(get_shot_model("physics"), TableShotModel)
    finally:
        registry._BACKENDS.clear()
        registry._BACKENDS.update(saved)


def _hole(cells: int = 50, res: float = 5.0) -> LieRaster:
    grid = np.full((cells, cells), _FW, dtype=np.int8)
    grid[cells // 2, cells // 2] = LIE_CODES["holed"]
    half = cells // 2 * res
    return LieRaster(
        codes=grid, origin=(-half, -half), resolution_yd=res, pin=(0, 0), tee=(0, 0)
    )


def _compare(
    profile, cells: int = 50, res: float = 5.0, n: int = 24, it: int = 40, **kw
):
    problem = StrategyProblem(
        _hole(cells, res),
        load_baseline(YAML),
        CourseConditions.benign(),
        ActionSet(clubs=("pw", "sw"), aim_grid_deg=np.array([-10.0, 0.0, 10.0])),
        State(x=-90.0, y=0.0, lie=_FW),
    )
    return compare_strategies(
        problem, profile, settings=SolveSettings(n_samples=n, max_iter=it), **kw
    )


def test_report_has_personalised_and_tour_strokes() -> None:
    rep = _compare(_profile(clubs={"pw": ClubSkill(skill_mult_lat=1.6)}))
    assert rep.tour_strokes > 0 and rep.personal_strokes > 0
    assert rep.delta == pytest.approx(rep.personal_strokes - rep.tour_strokes)
    assert rep.tour_action.club in ("pw", "sw")


def test_better_player_never_has_higher_expected_strokes() -> None:
    better = _compare(_profile(clubs={c: ClubSkill(0.8, 0.7) for c in ("pw", "sw")}))
    worse = _compare(_profile(clubs={c: ClubSkill(1.3, 1.6) for c in ("pw", "sw")}))
    assert better.personal_strokes <= worse.personal_strokes + 0.05
    assert better.delta <= worse.delta + 0.05


def _family_table(family: str):
    """Same carry for every family (centred), spread from the family's own draws."""

    def sim(impacts: np.ndarray, skill: str, wind: tuple[float, float]):
        along = 90.0 + (impacts[:, 0] - impacts[:, 0].mean())
        return np.stack([along, 3.0 * impacts[:, 1]], axis=1)

    return build_table(get_skill_family(family), sim, skills=("wedge",), n=2000, seed=1)


def _solve_with_table(family: str):
    model = TableShotModel(
        _family_table(family), get_shot_model("baseline", load_baseline(YAML))
    )
    # injected models use the scalar Bellman sweep, so keep the grid small
    return _compare(_profile(), cells=12, res=16.0, n=10, it=20, model=model)


def test_physics_backend_is_solved_and_tight_family_never_worse_than_loose() -> None:
    tight, loose = _solve_with_table("tour"), _solve_with_table("20_hcp")
    assert tight.personal_strokes > 0 and loose.personal_strokes > 0
    assert tight.personal_strokes <= loose.personal_strokes + 0.05
    # The table, not the baseline, drives the personal solve (tour solve is baseline).
    base = _compare(_profile(), cells=12, res=16.0, n=10, it=20)
    assert loose.personal_strokes != pytest.approx(base.personal_strokes, abs=1e-9)


def test_physics_path_ignores_profile_multipliers() -> None:
    """Skill enters via the table (fit_skill_family), not shrink_profile."""
    model = TableShotModel(_table(), get_shot_model("baseline", load_baseline(YAML)))
    args = (ShotAction("driver", 0.0), _FW, CourseConditions.benign())
    plain = model.sample(_profile(), *args, np.random.default_rng(3), 50)
    skewed = model.sample(
        _profile(clubs={"driver": ClubSkill(skill_mult_long=2.0)}),
        *args,
        np.random.default_rng(3),
        50,
    )
    np.testing.assert_array_equal(plain.along, skewed.along)
