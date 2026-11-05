"""Tour vs personalised strategy view: text and HTML/SVG renderers (#14572)."""

from __future__ import annotations

import html
from dataclasses import dataclass

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
    skipped); ``options`` then come from the tour baseline.
    """

    hole: str
    tour_strokes: float
    personal_strokes: float | None
    options: tuple[ShotOption, ...]

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
    model: ShotOutcomeModel | None = None,
    settings: SolveSettings = SolveSettings(),  # noqa: B008 - frozen, immutable
) -> StrategyView:
    """Solve the tour baseline and (unless ``quick_tour``) the player."""
    tour = PlayerProfile(name="tour", baseline=profile.baseline)
    t_mdp, t_val = solve_mdp(problem, tour, None, settings)
    tour_strokes = t_mdp.expected_strokes(problem.start, t_val)
    if quick_tour:
        opts = shot_options(t_mdp, t_val, problem.start, top=top)
        return StrategyView(hole, tour_strokes, None, opts)
    p_mdp, p_val = solve_mdp(problem, profile, model, settings)
    opts = shot_options(p_mdp, p_val, problem.start, model=model, top=top)
    return StrategyView(
        hole, tour_strokes, p_mdp.expected_strokes(problem.start, p_val), opts
    )


def _numbers(view: StrategyView) -> tuple[str, str, str]:
    personal = (
        "n/a (quick tour)"
        if view.personal_strokes is None
        else f"{view.personal_strokes:.2f}"
    )
    delta = "n/a" if view.delta is None else f"{view.delta:+.2f}"
    return f"{view.tour_strokes:.2f}", personal, delta


def render_text(view: StrategyView) -> str:
    """Plain-text report for the CLI."""
    tour, personal, delta = _numbers(view)
    lines = [
        f"Hole: {view.hole}",
        f"Expected strokes  tour: {tour}   personal: {personal}   delta: {delta}",
        "",
        f"{'club':<10}{'aim':>7}{'strokes':>9}{'carry':>8}{'95% long':>10}{'95% wide':>10}",
    ]
    for o in view.options:
        e = o.ellipse
        lines.append(
            f"{o.club:<10}{o.aim_deg:>6.1f}°{o.expected_strokes:>9.2f}"
            f"{e.centre_along:>8.0f}{e.semi_major:>10.1f}{e.semi_minor:>10.1f}"
        )
    return "\n".join(lines) + "\n"


def render_html(view: StrategyView) -> str:
    """Self-contained page: both numbers, option table and an SVG of the ellipses."""
    tour, personal, delta = _numbers(view)
    reach = max(o.ellipse.centre_along + o.ellipse.semi_major for o in view.options)
    span = max(o.ellipse.semi_major for o in view.options) * 1.2
    shapes = "".join(
        f'<ellipse cx="{o.ellipse.centre_along:.1f}" cy="{o.ellipse.centre_lateral:.1f}" '
        f'rx="{o.ellipse.semi_major:.1f}" ry="{o.ellipse.semi_minor:.1f}" '
        f'transform="rotate({-o.ellipse.angle_deg:.1f} {o.ellipse.centre_along:.1f} {o.ellipse.centre_lateral:.1f})" '
        f'fill="rgba(31,119,180,0.15)" stroke="#1f77b4"><title>{html.escape(o.club)}</title></ellipse>'
        f'<text x="{o.ellipse.centre_along:.1f}" y="{o.ellipse.centre_lateral:.1f}" font-size="6" '
        f'text-anchor="middle">{html.escape(o.club)}</text>'
        for o in view.options
    )
    rows = "".join(
        f"<tr><td>{html.escape(o.club)}</td><td>{o.aim_deg:.1f}&deg;</td>"
        f"<td>{o.expected_strokes:.2f}</td></tr>"
        for o in view.options
    )
    return (
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>Strategy: {html.escape(view.hole)}</title>"
        f"<h1>{html.escape(view.hole)}</h1>"
        f"<p>Tour SG <b>{tour}</b> &middot; Personal SG <b>{personal}</b> &middot; Delta <b>{delta}</b></p>"
        f"<svg viewBox='-10 {-span:.0f} {reach + 20:.0f} {2 * span:.0f}' width='100%' "
        f"role='img' aria-label='Dispersion ellipses'>{shapes}</svg>"
        f"<table><tr><th>Club</th><th>Aim</th><th>Strokes</th></tr>{rows}</table>"
    )
