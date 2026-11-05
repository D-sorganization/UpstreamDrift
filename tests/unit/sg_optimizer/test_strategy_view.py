"""Shot options, dispersion ellipses and the tour/personal strategy view (#14572)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LieRaster
from src.shared.python.sg_optimizer.mdp.action import ActionSet
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.outcome import LandingSamples
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile
from src.shared.python.sg_optimizer.shot_options import (
    dispersion_ellipse,
    shot_options,
)
from src.shared.python.sg_optimizer.strategy import (
    SolveSettings,
    StrategyProblem,
    compare_strategies,
    solve_mdp,
)
from src.shared.python.sg_optimizer.strategy_view import (
    build_view,
    render_html,
    render_text,
)

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[3] / "data/sg_optimizer/baselines/pga_tour.yaml"
_FW = LIE_CODES["fairway"]
CFG = SolveSettings(n_samples=24, max_iter=40)


def _problem(clubs: tuple[str, ...] = ("driver", "pw")) -> StrategyProblem:
    cells, res = 50, 5.0
    grid = np.full((cells, cells), _FW, dtype=np.int8)
    grid[cells // 2, cells // 2] = LIE_CODES["holed"]
    half = cells // 2 * res
    raster = LieRaster(
        codes=grid, origin=(-half, -half), resolution_yd=res, pin=(0, 0), tee=(0, 0)
    )
    return StrategyProblem(
        raster,
        load_baseline(YAML),
        CourseConditions.benign(),
        ActionSet(clubs=clubs, aim_grid_deg=np.array([-10.0, 0.0, 10.0])),
        State(x=-100.0, y=0.0, lie=_FW),
    )


def _profile() -> PlayerProfile:
    return PlayerProfile(name="p", baseline=str(YAML))


def test_ellipse_recovers_known_sigmas_and_coverage() -> None:
    rng = np.random.default_rng(0)
    s = LandingSamples(
        along=rng.normal(150, 30, 20000), lateral=rng.normal(2, 10, 20000)
    )
    e = dispersion_ellipse(s, coverage=0.95)
    k = np.sqrt(-2 * np.log(0.05))
    assert e.semi_major == pytest.approx(k * 30, rel=0.03)
    assert e.semi_minor == pytest.approx(k * 10, rel=0.03)
    assert e.centre_along == pytest.approx(150, abs=1.0)
    assert abs(e.angle_deg) < 3.0
    dx = (s.along - e.centre_along) / e.semi_major
    dy = (s.lateral - e.centre_lateral) / e.semi_minor
    assert np.mean(dx**2 + dy**2 <= 1.0) == pytest.approx(0.95, abs=0.01)


def test_ellipse_rejects_bad_coverage_and_single_sample() -> None:
    s = LandingSamples(
        along=np.array([1.0, 2.0, 3.0]), lateral=np.array([0.0, 1.0, 0.5])
    )
    with pytest.raises(ContractViolationError):
        dispersion_ellipse(s, coverage=1.0)
    one = LandingSamples(along=np.array([1.0]), lateral=np.array([0.0]))
    with pytest.raises(ContractViolationError):
        dispersion_ellipse(one)


def test_action_values_argmin_is_the_optimal_action() -> None:
    mdp, value = solve_mdp(_problem(), _profile(), None, CFG)
    q = mdp.action_values(_problem().start, value)
    best = min(q, key=lambda av: av[1])[0]
    assert best == mdp.optimal_action(_problem().start, value)


def test_shot_options_one_per_club_sorted_best_first() -> None:
    prob = _problem()
    mdp, value = solve_mdp(prob, _profile(), None, CFG)
    opts = shot_options(mdp, value, prob.start, top=5)
    assert [o.club for o in opts].count("driver") == 1
    assert len(opts) == 2
    assert opts[0].expected_strokes <= opts[1].expected_strokes
    best_q = min(q for _, q in mdp.action_values(prob.start, value))
    assert opts[0].expected_strokes == pytest.approx(best_q)
    assert shot_options(mdp, value, prob.start, top=1)[0].club == opts[0].club


def test_driver_ellipse_is_larger_than_wedge_ellipse() -> None:
    prob = _problem()
    mdp, value = solve_mdp(prob, _profile(), None, CFG)
    by_club = {o.club: o for o in shot_options(mdp, value, prob.start, top=5)}
    assert by_club["driver"].ellipse.semi_major > by_club["pw"].ellipse.semi_major
    assert by_club["driver"].ellipse.centre_along > by_club["pw"].ellipse.centre_along


def test_view_reports_tour_and_personal_matching_compare_strategies() -> None:
    prob, prof = _problem(), _profile()
    view = build_view("synthetic", prob, prof, settings=CFG)
    ref = compare_strategies(prob, prof, settings=CFG)
    assert view.tour_strokes == pytest.approx(ref.tour_strokes)
    assert view.personal_strokes == pytest.approx(ref.personal_strokes)
    assert view.delta == pytest.approx(ref.delta)
    assert view.options


def test_quick_tour_solves_the_tour_only() -> None:
    prob, prof = _problem(), _profile()
    view = build_view("synthetic", prob, prof, quick_tour=True, settings=CFG)
    ref = compare_strategies(prob, prof, settings=CFG)
    assert view.personal_strokes is None and view.delta is None
    assert view.tour_strokes == pytest.approx(ref.tour_strokes)
    assert "n/a" in render_text(view)


def test_text_and_html_show_both_numbers_and_ellipses() -> None:
    view = build_view("synthetic", _problem(), _profile(), settings=CFG)
    text, html = render_text(view), render_html(view)
    for token in (f"{view.tour_strokes:.2f}", f"{view.personal_strokes:.2f}"):
        assert token in text and token in html
    assert "tour" in text.lower() and "personal" in text.lower()
    assert html.count("<ellipse") == len(view.options)
    assert view.options[0].club in html
