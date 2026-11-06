"""``strategy`` sub-command: tour vs personalised strategy at the tee (#14572)."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from src.shared.python.contracts import require

from src.shared.python.sg_optimizer.course.rasterize import rasterize_synthetic
from src.shared.python.sg_optimizer.mdp.action import default_action_set
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.shot_model.baseline import (
    BaselineBag,
    load_baseline,
)
from src.shared.python.sg_optimizer.shot_model.baseline_model import BaselineShotModel
from src.shared.python.sg_optimizer.shot_model.dispersion_table import DispersionTable
from src.shared.python.sg_optimizer.shot_model.outcome import ShotOutcomeModel
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile
from src.shared.python.sg_optimizer.shot_model.registry import get_shot_model
from src.shared.python.sg_optimizer.shot_model.table_model import TableShotModel
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
    model = p.add_mutually_exclusive_group()
    model.add_argument(
        "--shot-table",
        metavar="PATH",
        help="dispersion-table JSON for the personal solve (tour baseline for "
        "clubs the table does not cover)",
    )
    model.add_argument(
        "--shot-model", metavar="NAME", help="registered shot-model backend"
    )
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

    baseline = load_baseline(args.baseline)
    model, label = _shot_model(args, baseline)
    hole = (
        _load_hole_spec(args.hole_spec)
        if args.hole_spec is not None
        else _classic_to_synthetic(args.classic)
    )
    raster = rasterize_synthetic(hole, resolution_yd=args.resolution)
    if not args.pin_only:
        raster = green_terminal(raster)
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
        green_terminal_mode=not args.pin_only,
        model=model,
        settings=SolveSettings(args.n_samples, args.seed, args.max_iter),
    )
    view = dataclasses.replace(view, shot_model=label)
    sys.stdout.write(render_text(view))
    if args.html:
        Path(args.html).write_text(render_html(view), encoding="utf-8")
    return 0


def _shot_model(
    args: argparse.Namespace, baseline: BaselineBag
) -> tuple[ShotOutcomeModel | None, str]:
    """The explicitly selected model and its provenance label (fail closed).

    No selection keeps the default baseline (``None``). An explicit selection is
    validated before any solve and never replaced by the baseline.
    """
    explicit = args.shot_table is not None or args.shot_model is not None
    require(
        not (explicit and args.quick_tour),
        "--quick-tour skips the personal solve; drop --shot-table/--shot-model",
    )
    if args.shot_table is not None:
        path = Path(args.shot_table)
        require(path.is_file(), f"shot table not found: {path}")
        table = DispersionTable.from_json(path)
        model = TableShotModel(table, BaselineShotModel(baseline))
        return model, f"table {table.table_hash} ({path.name})"
    if args.shot_model is not None:
        return get_shot_model(args.shot_model, baseline), args.shot_model
    return None, "baseline"
