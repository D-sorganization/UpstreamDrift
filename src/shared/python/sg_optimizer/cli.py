"""Headless end-to-end CLI for the Strokes Gained Optimizer.

Usage (synthetic hole spec, Phase 1)::

    python -m src.shared.python.sg_optimizer.cli \
        --profile path/to/player.yaml \
        --hole-spec path/to/hole.py \
        --conditions tournament \
        --baseline data/sg_optimizer/baselines/pga_tour.yaml

Usage (classic hole, Phase 2)::

    python -m src.shared.python.sg_optimizer.cli run \
        --classic sawgrass_17 \
        --profile path/to/player.yaml \
        --baseline data/sg_optimizer/baselines/pga_tour.yaml

``--hole-spec`` is a Python file exporting a top-level ``HOLE`` SyntheticHole
object.
``--classic`` is one of the slugs returned by ``list_classics()``.
``--conditions`` is one of {``benign``, ``tournament``, ``major``} or a YAML
path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import (
    CircleFeature,
    LIE_CODES,
    RectFeature,
    SyntheticHole,
    rasterize_synthetic,
)
from src.shared.python.sg_optimizer.mdp.action import default_action_set
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.mdp.value_iteration import HoleMDP
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile


_CONDITION_PRESETS = {
    "benign": CourseConditions.benign,
    "tournament": CourseConditions.tournament,
    "major": CourseConditions.major_championship,
}


def _load_conditions(spec: str) -> CourseConditions:
    if spec in _CONDITION_PRESETS:
        return _CONDITION_PRESETS[spec]()
    return CourseConditions.from_yaml(spec)


def _load_hole_spec(path: str) -> SyntheticHole:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location("_hole_spec", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import hole spec at {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hole = getattr(module, "HOLE", None)
    if not isinstance(hole, SyntheticHole):
        raise TypeError(f"hole spec {p} must export top-level HOLE: SyntheticHole")
    return hole


def _classic_to_synthetic(slug: str) -> SyntheticHole:
    """Load a classic HoleGeometry and convert to a SyntheticHole for the MDP.

    The conversion uses the GeoJSON yardage as the hole length and constructs
    a simplified rectangular layout so the existing raster-based MDP can run
    without a full georeferenced rasterizer (Phase 2 spec §2.4).

    Coordinate frame: tee at (0, 0), pin at (yardage, 0).
    """
    from src.shared.python.sg_optimizer.course.library import load_classic  # noqa: PLC0415

    geom = load_classic(slug)
    length = float(geom.yardage)
    half_w = 25.0 if geom.par >= 5 else 20.0 if geom.par == 4 else 15.0

    # Determine whether there is a water hazard in the GeoJSON.
    has_water = len(geom.water) > 0
    has_bunker = len(geom.bunker) > 0

    features: list[RectFeature | CircleFeature] = []

    if geom.par >= 4:
        # Fairway corridor along the centreline.
        fw_w = 18.0 if geom.par == 4 else 22.0
        features.append(
            RectFeature("fairway", 30.0, length - 15.0, -fw_w / 2, fw_w / 2)
        )

    # Green circle at the far end.
    green_r = 11.0 if geom.par >= 5 else 9.0
    features.append(CircleFeature("green", length, 0.0, green_r))

    if has_water:
        # Simplified water hazard along the right side.
        features.append(RectFeature("water", 40.0, length - 20.0, -half_w, -half_w / 2))

    if has_bunker:
        # Greenside bunker.
        features.append(CircleFeature("sand", length - 5.0, half_w / 2, 5.0))

    tee_x, tee_y = 0.0, 0.0
    pin_x, pin_y = length, 0.0

    return SyntheticHole(
        name=geom.name,
        par=geom.par,
        tee=(tee_x, tee_y),
        pin=(pin_x, pin_y),
        bbox=(-20.0, length + 20.0, -half_w - 5.0, half_w + 5.0),
        features=tuple(features),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sg-optimizer")
    subparsers = parser.add_subparsers(dest="command")

    # --- ``run`` sub-command (Phase 1 + Phase 2) --------------------------
    run_p = subparsers.add_parser("run", help="Solve the hole MDP and report strategy")
    run_p.add_argument("--profile", required=True)
    run_p.add_argument("--baseline", required=True)
    run_p.add_argument("--conditions", default="tournament")
    run_p.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="raster resolution in yards (default 2.0)",
    )
    run_p.add_argument("--n-samples", type=int, default=48)
    run_p.add_argument("--max-iter", type=int, default=80)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument(
        "--from-tee",
        action="store_true",
        help="report optimal action from the tee state",
    )

    hole_group = run_p.add_mutually_exclusive_group(required=True)
    hole_group.add_argument(
        "--hole-spec",
        help="Python file exporting top-level HOLE: SyntheticHole",
    )
    hole_group.add_argument(
        "--classic",
        metavar="SLUG",
        help="classic hole slug (e.g. sawgrass_17); see list-classics",
    )

    # --- ``list-classics`` sub-command (Phase 2) ---------------------------
    subparsers.add_parser("list-classics", help="List available classic hole slugs")

    args = parser.parse_args(argv)

    # Handle legacy invocation: no sub-command → treat as legacy ``run``.
    if args.command is None:
        # Re-parse with a legacy flat parser for backward compatibility.
        return _legacy_main(argv)

    if args.command == "list-classics":
        from src.shared.python.sg_optimizer.course.library import list_classics  # noqa: PLC0415

        for slug in list_classics():
            sys.stdout.write(slug + "\n")
        return 0

    # --- ``run`` -----------------------------------------------------------
    profile = PlayerProfile.from_yaml(args.profile)
    baseline = load_baseline(args.baseline)
    conditions = _load_conditions(args.conditions)

    if args.hole_spec is not None:
        hole = _load_hole_spec(args.hole_spec)
    else:
        hole = _classic_to_synthetic(args.classic)

    raster = rasterize_synthetic(hole, resolution_yd=args.resolution)

    mdp = HoleMDP(
        raster=raster,
        profile=profile,
        baseline=baseline,
        conditions=conditions,
        actions=default_action_set(),
        n_samples=args.n_samples,
        seed=args.seed,
    )
    result = mdp.solve(max_iter=args.max_iter)

    tee_state = State(x=hole.tee[0], y=hole.tee[1], lie=int(raster.lie_at(*hole.tee)))
    optimal = mdp.optimal_action(tee_state, result.value)
    expected = mdp.expected_strokes(tee_state, result.value)

    payload = {
        "hole": hole.name,
        "par": hole.par,
        "profile": profile.name,
        "conditions": args.conditions,
        "iterations": result.iterations,
        "delta": result.delta,
        "tee_expected_strokes": expected if math.isfinite(expected) else None,
        "tee_optimal_action": {
            "club": optimal.club,
            "aim_angle_deg": math.degrees(optimal.aim_angle_rad),
        },
    }
    json.dump(payload, sys.stdout, indent=2, default=float)
    sys.stdout.write("\n")
    return 0


def _legacy_main(argv: list[str] | None) -> int:
    """Backward-compatible parser for the Phase-1 flat CLI invocation."""
    parser = argparse.ArgumentParser(prog="sg-optimizer")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--hole-spec", required=True)
    parser.add_argument("--conditions", default="tournament")
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from-tee", action="store_true")
    args = parser.parse_args(argv)

    profile = PlayerProfile.from_yaml(args.profile)
    baseline = load_baseline(args.baseline)
    conditions = _load_conditions(args.conditions)
    hole = _load_hole_spec(args.hole_spec)
    raster = rasterize_synthetic(hole, resolution_yd=args.resolution)

    mdp = HoleMDP(
        raster=raster,
        profile=profile,
        baseline=baseline,
        conditions=conditions,
        actions=default_action_set(),
        n_samples=args.n_samples,
        seed=args.seed,
    )
    result = mdp.solve(max_iter=args.max_iter)

    tee_state = State(x=hole.tee[0], y=hole.tee[1], lie=int(raster.lie_at(*hole.tee)))
    optimal = mdp.optimal_action(tee_state, result.value)
    expected = mdp.expected_strokes(tee_state, result.value)

    payload = {
        "hole": hole.name,
        "par": hole.par,
        "profile": profile.name,
        "conditions": args.conditions,
        "iterations": result.iterations,
        "delta": result.delta,
        "tee_expected_strokes": expected if math.isfinite(expected) else None,
        "tee_optimal_action": {
            "club": optimal.club,
            "aim_angle_deg": math.degrees(optimal.aim_angle_rad),
        },
    }
    json.dump(payload, sys.stdout, indent=2, default=float)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# Silence unused-import in env without numpy plot deps.
_ = np
# Silence unused imports brought in for _classic_to_synthetic type annotations.
_lie_codes_ref = LIE_CODES
