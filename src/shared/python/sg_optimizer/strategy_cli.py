"""``strategy`` sub-command: tour vs personalised strategy at the tee (#14572)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.shared.python.sg_optimizer.course.rasterize import rasterize_synthetic
from src.shared.python.sg_optimizer.mdp.action import default_action_set
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile
from src.shared.python.sg_optimizer.strategy import (
    SolveSettings,
    StrategyProblem,
    green_terminal,
)
from src.shared.python.sg_optimizer.strategy_view import (
    build_view,
    render_html,
    render_text,
)


def add_strategy_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``strategy`` on the main CLI's sub-parsers."""
    p = subparsers.add_parser(
        "strategy", help="Tour vs personal strokes + shot options"
    )
    p.add_argument("--profile", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--conditions", default="tournament")
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--n-samples", type=int, default=48)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=5, help="shot options to list")
    p.add_argument(
        "--quick-tour",
        action="store_true",
        help="tour baseline only: skip the personalised solve",
    )
    p.add_argument(
        "--pin-only",
        action="store_true",
        help="end the hole only at the pin cell (default: any green cell, i.e. "
        "strokes to reach the green; pin-only targets one cell, so its value "
        "depends on the raster resolution)",
    )
    p.add_argument("--html", metavar="PATH", help="also write the HTML/SVG view")
    hole = p.add_mutually_exclusive_group(required=True)
    hole.add_argument("--hole-spec")
    hole.add_argument("--classic", metavar="SLUG")


def run_strategy(args: argparse.Namespace) -> int:
    """Solve the hole and print (and optionally save) the strategy view."""
    from src.shared.python.sg_optimizer.cli import (  # noqa: PLC0415
        _classic_to_synthetic,
        _load_conditions,
        _load_hole_spec,
    )

    hole = (
        _load_hole_spec(args.hole_spec)
        if args.hole_spec is not None
        else _classic_to_synthetic(args.classic)
    )
    raster = rasterize_synthetic(hole, resolution_yd=args.resolution)
    if not args.pin_only:
        raster = green_terminal(raster)
    baseline = load_baseline(args.baseline)
    problem = StrategyProblem(
        raster,
        baseline,
        _load_conditions(args.conditions),
        default_action_set(),
        State(x=hole.tee[0], y=hole.tee[1], lie=int(raster.lie_at(*hole.tee))),
    )
    view = build_view(
        hole.name,
        problem,
        PlayerProfile.from_yaml(args.profile),
        quick_tour=args.quick_tour,
        top=args.top,
        settings=SolveSettings(args.n_samples, args.seed, args.max_iter),
    )
    sys.stdout.write(render_text(view))
    if args.html:
        Path(args.html).write_text(render_html(view), encoding="utf-8")
    return 0
