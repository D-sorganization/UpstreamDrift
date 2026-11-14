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
    DispersionEllipse,
    ShotOption,
    dispersion_ellipse,
    shot_options,
)
from src.shared.python.sg_optimizer.strategy import (
    SolveSettings,
    StrategyProblem,
    compare_strategies,
    green_terminal,
    solve_mdp,
)
from src.shared.python.sg_optimizer.strategy_view import (
    StrategyView,
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
    mdp, res = solve_mdp(_problem(), _profile(), None, CFG)
    value = res.value
    q = mdp.action_values(_problem().start, value)
    best = min(q, key=lambda av: av[1])[0]
    assert best == mdp.optimal_action(_problem().start, value)


def test_shot_options_one_per_club_sorted_best_first() -> None:
    prob = _problem()
    mdp, res = solve_mdp(prob, _profile(), None, CFG)
    value = res.value
    opts = shot_options(mdp, value, prob.start, top=5)
    assert [o.club for o in opts].count("driver") == 1
    assert len(opts) == 2
    assert opts[0].expected_strokes <= opts[1].expected_strokes
    best_q = min(q for _, q in mdp.action_values(prob.start, value))
    assert opts[0].expected_strokes == pytest.approx(best_q)
    assert shot_options(mdp, value, prob.start, top=1)[0].club == opts[0].club


def test_driver_ellipse_is_larger_than_wedge_ellipse() -> None:
    prob = _problem()
    mdp, res = solve_mdp(prob, _profile(), None, CFG)
    value = res.value
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


def test_green_terminal_ends_the_hole_on_every_green_cell_only() -> None:
    raster = _problem().raster
    codes = raster.codes.copy()
    codes[10:14, 10:14] = LIE_CODES["green"]
    src = LieRaster(codes, raster.resolution_yd, raster.origin, raster.pin, raster.tee)
    out = green_terminal(src)
    holed = LIE_CODES["holed"]
    assert np.all(out.codes[10:14, 10:14] == holed)
    assert np.sum(out.codes == holed) == np.sum(src.codes == holed) + 16
    assert np.sum(out.codes != src.codes) == 16
    assert src.codes[10, 10] == LIE_CODES["green"]  # input untouched


def test_view_states_sample_count_and_convergence() -> None:
    view = build_view("synthetic", _problem(), _profile(), settings=CFG)
    assert view.n_samples == CFG.n_samples and 0 < view.iterations <= CFG.max_iter
    assert "MC n=24" in render_text(view)
    assert view.converged == (view.residual < 1e-3)
    assert ("NOT CONVERGED" in render_text(view)) == (not view.converged)


def test_aims_are_labelled_as_relative_to_the_pin_line() -> None:
    """Solver aims are pin-relative (0 = at the pin); the report must say so."""
    view = build_view("synthetic", _problem(), _profile(), settings=CFG)
    assert "aim vs pin" in render_text(view)
    assert "from pin line" in render_html(view)


def _demo_par4() -> StrategyProblem:
    """Fairway, water right of it and a green 380 yd out; full club set."""
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
    raster = green_terminal(rasterize_synthetic(hole, resolution_yd=20.0))
    return StrategyProblem(
        raster,
        load_baseline(YAML),
        CourseConditions.benign(),
        default_action_set(),
        State(x=0.0, y=0.0, lie=int(raster.lie_at(0.0, 0.0))),
    )


def test_reference_par4_converges_and_says_so() -> None:
    """The B8 demo hole reaches its stated tolerance, so the report may say so."""
    cfg = SolveSettings(n_samples=24, max_iter=80)
    view = build_view("par4", _demo_par4(), _profile(), settings=cfg)
    assert view.converged and view.residual < 1e-3 and view.iterations < 80
    assert 1.0 < view.tour_strokes < 5.0  # strokes to reach the green
    assert "NOT CONVERGED" not in render_text(view)
    assert "NOT CONVERGED" not in render_html(view)


def test_header_says_strokes_to_green_not_strokes_gained() -> None:
    """The numbers are strokes to reach the green; "SG" would misname them."""
    view = build_view("par4", _problem(), _profile(), settings=CFG)
    text, page = render_text(view), render_html(view)
    for out in (text, page):
        assert "SG" not in out
        assert "strokes to green" in out.lower()
        assert "green ends the hole" in out
    assert "personal − tour" in page.lower().replace("&minus;", "−")


def test_pin_only_mode_is_stated() -> None:
    view = build_view(
        "par4", _problem(), _profile(), settings=CFG, green_terminal_mode=False
    )
    assert "pin cell only" in render_text(view)
    assert "pin cell only" in render_html(view)


@pytest.mark.parametrize(
    ("green_terminal", "expected_headline", "terminal_note"),
    [
        (
            True,
            "Expected strokes to green",
            "green ends the hole (putting is not modelled)",
        ),
        (
            False,
            "Expected strokes to pin cell",
            "pin cell only (value depends on raster resolution)",
        ),
    ],
    ids=("green-terminal", "pin-only"),
)
def test_text_and_html_headlines_match_the_terminal(
    green_terminal: bool, expected_headline: str, terminal_note: str
) -> None:
    base = _one_option_view(0.0)
    view = StrategyView(
        base.hole,
        base.tour_strokes,
        base.personal_strokes,
        base.options,
        green_terminal=green_terminal,
    )
    outputs = (render_text(view), render_html(view))
    missing = [
        surface
        for surface, output in zip(("text", "HTML"), outputs, strict=True)
        if expected_headline not in output or terminal_note not in output
    ]
    assert not missing, f"wrong terminal headline or note in: {missing}"


def _capped_tour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the tour solve report a large residual; the player solve is real."""
    import dataclasses

    from src.shared.python.sg_optimizer import strategy_view as sv

    real = sv.solve_mdp

    def fake(problem, profile, model, cfg):  # type: ignore[no-untyped-def]
        mdp, res = real(problem, profile, model, cfg)
        if profile.name == "tour":
            res = dataclasses.replace(res, delta=0.5, iterations=cfg.max_iter)
        return mdp, res

    monkeypatch.setattr(sv, "solve_mdp", fake)


@pytest.mark.parametrize("quick", [False, True])
def test_unconverged_tour_solve_is_reported(
    monkeypatch: pytest.MonkeyPatch, quick: bool
) -> None:
    _capped_tour(monkeypatch)
    view = build_view("p", _problem(), _profile(), quick_tour=quick, settings=CFG)
    assert not view.converged and view.residual == pytest.approx(0.5)
    assert "NOT CONVERGED" in render_text(view)
    assert "NOT CONVERGED" in render_html(view)


def test_convergence_tolerance_comes_from_the_solver_settings() -> None:
    loose = SolveSettings(n_samples=24, max_iter=40, tol=1e3)
    view = build_view("p", _problem(), _profile(), settings=loose)
    assert view.converged and view.iterations == 1


def _one_option_view(aim_deg: float, **ell: float) -> StrategyView:
    ellipse = DispersionEllipse(
        centre_along=ell.get("along", 100.0),
        centre_lateral=ell.get("lateral", 0.0),
        semi_major=20.0,
        semi_minor=5.0,
        angle_deg=ell.get("angle", 0.0),
    )
    opt = ShotOption(
        club="7_iron", aim_deg=aim_deg, expected_strokes=3.0, ellipse=ellipse
    )
    return StrategyView("h", 3.0, 3.2, (opt,))


def test_ellipses_are_drawn_in_the_golfers_pin_line_frame() -> None:
    """Aimed 30 deg left of the pin, 100 yd out: centre at (86.6, 50) up the page."""
    page = render_html(_one_option_view(30.0))
    assert 'cx="86.6" cy="-50.0"' in page
    assert "rotate(-30.0 86.6 -50.0)" in page  # axis 0 + aim 30, y flipped


def test_tilted_cloud_rotation_and_lateral_signs() -> None:
    """A left-of-aim offset plots up the page; a CCW-tilted axis rotates negative."""
    page = render_html(_one_option_view(0.0, along=100.0, lateral=10.0, angle=20.0))
    assert 'cx="100.0" cy="-10.0"' in page
    assert "rotate(-20.0 100.0 -10.0)" in page
    right = render_html(_one_option_view(-10.0, along=100.0, angle=-5.0))
    assert "rotate(15.0 " in right  # -5 + -10 = -15 CCW -> +15 in SVG


def test_page_states_the_frame() -> None:
    assert "pin line" in render_html(_one_option_view(0.0)).lower()
