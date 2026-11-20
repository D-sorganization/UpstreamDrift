"""Tour vs personalised strategy view: text and HTML/SVG renderers (#14572)."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

from src.shared.python.sg_optimizer.mdp.value_iteration import SolveResult
from src.shared.python.sg_optimizer.shot_model.outcome import ShotOutcomeModel
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile
from src.shared.python.sg_optimizer.shot_options import ShotOption, shot_options
from src.shared.python.sg_optimizer.strategy import (
    SolveSettings,
    StrategyProblem,
    solve_mdp,
)


@dataclass(frozen=True)
class StrategyView:
    """Both expected-strokes numbers plus the shot options for one scenario.

    ``personal_strokes`` is ``None`` in quick-tour mode (the player solve is
    skipped); ``options`` then come from the tour baseline. ``shot_model`` names
    the model behind the personal solve (#14744).
    """

    hole: str
    tour_strokes: float
    personal_strokes: float | None
    options: tuple[ShotOption, ...]
    n_samples: int = 0
    iterations: int = 0
    residual: float = 0.0
    converged: bool = True
    green_terminal: bool = True
    shot_model: str = "baseline"

    @property
    def delta(self) -> float | None:
        """Personalised minus tour expected strokes (positive = worse than tour)."""
        if self.personal_strokes is None:
            return None
        return self.personal_strokes - self.tour_strokes


def build_view(
    hole: str,
    problem: StrategyProblem,
    profile: PlayerProfile,
    *,
    quick_tour: bool = False,
    top: int = 5,
    green_terminal_mode: bool = True,
    model: ShotOutcomeModel | None = None,
    settings: SolveSettings = SolveSettings(),  # noqa: B008 - frozen, immutable
) -> StrategyView:
    """Solve the tour baseline and (unless ``quick_tour``) the player."""
    tour = PlayerProfile(name="tour", baseline=profile.baseline)
    t_mdp, t_res = solve_mdp(problem, tour, None, settings)
    tour_strokes = t_mdp.expected_strokes(problem.start, t_res.value)
    results: tuple[SolveResult, ...]
    personal: float | None
    if quick_tour:
        opts = shot_options(t_mdp, t_res.value, problem.start, top=top)
        results = (t_res,)
        personal = None
    else:
        p_mdp, p_res = solve_mdp(problem, profile, model, settings)
        opts = shot_options(p_mdp, p_res.value, problem.start, model=model, top=top)
        personal = p_mdp.expected_strokes(problem.start, p_res.value)
        results = (t_res, p_res)
    return _view(
        hole, tour_strokes, personal, opts, results, settings, green_terminal_mode
    )


def _view(
    hole: str,
    tour: float,
    personal: float | None,
    options: tuple[ShotOption, ...],
    results: tuple[SolveResult, ...],
    settings: SolveSettings,
    green_terminal_mode: bool,
) -> StrategyView:
    """Fold every solve into one verdict: the worst residual decides convergence."""
    residual = max(r.delta for r in results)
    return StrategyView(
        hole,
        tour,
        personal,
        options,
        n_samples=settings.n_samples,
        iterations=max(r.iterations for r in results),
        residual=residual,
        converged=residual < settings.tol,
        green_terminal=green_terminal_mode,
    )


def _numbers(view: StrategyView) -> tuple[str, str, str]:
    personal = (
        "n/a (quick tour)"
        if view.personal_strokes is None
        else f"{view.personal_strokes:.2f}"
    )
    delta = "n/a" if view.delta is None else f"{view.delta:+.2f}"
    return f"{view.tour_strokes:.2f}", personal, delta


def _solve_note(view: StrategyView) -> str:
    state = "converged" if view.converged else "NOT CONVERGED"
    return (
        f"MC n={view.n_samples} per action, {view.iterations} sweeps, "
        f"{state} (max change {view.residual:.3g})"
    )


def _terminal_note(view: StrategyView) -> str:
    if view.green_terminal:
        return "Terminal: reaching the green ends the hole (putting is not modelled)"
    return "Terminal: pin cell only (value depends on raster resolution)"


def _terminal_target_name(view: StrategyView) -> str:
    """Name the endpoint whose expected-strokes value is being reported."""
    return "green" if view.green_terminal else "pin cell"


def _pin_frame(o: ShotOption) -> tuple[float, float, float]:
    """Ellipse centre and axis in the golfer's pin-line frame, SVG coordinates.

    The option's ellipse is in its own aim frame; rotating by ``aim_deg`` puts
    every option on one common x-axis (the pin line, pin to the right). SVG y runs
    down, so left of the pin line plots up and rotations flip sign.
    """
    e, a = o.ellipse, math.radians(o.aim_deg)
    x = e.centre_along * math.cos(a) - e.centre_lateral * math.sin(a)
    y = e.centre_along * math.sin(a) + e.centre_lateral * math.cos(a)
    return x, -y, -(e.angle_deg + o.aim_deg)


def render_text(view: StrategyView) -> str:
    """Plain-text report for the CLI."""
    tour, personal, delta = _numbers(view)
    lines = [
        f"Hole: {view.hole}",
        f"Expected strokes to {_terminal_target_name(view)}  "
        f"tour: {tour}   personal: {personal}   personal - tour: {delta}",
        _terminal_note(view),
        _solve_note(view),
        f"Shot model: {view.shot_model}",
        "",
        f"{'club':<10}{'aim vs pin':>11}{'strokes':>9}{'carry':>8}{'95% long':>10}{'95% wide':>10}",
    ]
    for o in view.options:
        e = o.ellipse
        lines.append(
            f"{o.club:<10}{o.aim_deg:>10.1f}°{o.expected_strokes:>9.2f}"
            f"{e.centre_along:>8.0f}{e.semi_major:>10.1f}{e.semi_minor:>10.1f}"
        )
    return "\n".join(lines) + "\n"


def render_html(view: StrategyView) -> str:
    """Self-contained page: both numbers, option table and an SVG of the ellipses."""
    tour, personal, delta = _numbers(view)
    placed = [(_pin_frame(o), o) for o in view.options]
    far = max(x + o.ellipse.semi_major for (x, _, _), o in placed)
    near = min(x - o.ellipse.semi_major for (x, _, _), o in placed)
    span = max(abs(y) + o.ellipse.semi_major for (_, y, _), o in placed)
    shapes = "".join(
        f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{o.ellipse.semi_major:.1f}" '
        f'ry="{o.ellipse.semi_minor:.1f}" transform="rotate({rot:.1f} {x:.1f} {y:.1f})" '
        f'fill="rgba(31,119,180,0.15)" stroke="#1f77b4"><title>{html.escape(o.club)}</title></ellipse>'
        f'<text x="{x:.1f}" y="{y - o.ellipse.semi_minor - 2 - 7 * (i % 2):.1f}" font-size="6" '
        f'text-anchor="middle">{html.escape(o.club)}</text>'
        for i, ((x, y, rot), o) in enumerate(placed)
    )
    rows = "".join(
        f"<tr><td>{html.escape(o.club)}</td><td>{o.aim_deg:.1f}&deg;</td>"
        f"<td>{o.expected_strokes:.2f}</td></tr>"
        for o in view.options
    )
    return (
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>Strategy: {html.escape(view.hole)}</title>"
        f"<h1>{html.escape(view.hole)}</h1><p>{html.escape(_solve_note(view))}</p>"
        f"<p>{html.escape(_terminal_note(view))}</p>"
        f"<p>Shot model: {html.escape(view.shot_model)}</p>"
        f"<p>Expected strokes to {_terminal_target_name(view)}: tour <b>{tour}</b> &middot; personal <b>{personal}</b>"
        f" &middot; personal &minus; tour <b>{delta}</b></p>"
        f"<svg viewBox='{near - 10:.0f} {-span - 4:.0f} {far - near + 20:.0f} {2 * span + 8:.0f}' "
        "width='100%' style='max-height:70vh' role='img' aria-label='Dispersion ellipses'>"
        f"{shapes}</svg>"
        "<p>95% landing ellipses seen from the tee along the pin line "
        "(pin to the right, left of the line up the page).</p>"
        f"<table><tr><th>Club</th><th>Aim (from pin line)</th><th>Strokes</th></tr>{rows}</table>"
    )
