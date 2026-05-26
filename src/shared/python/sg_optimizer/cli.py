"""Headless end-to-end CLI for the Phase-1 Strokes Gained Optimizer.

Usage::

    python -m src.shared.python.sg_optimizer.cli \
        --profile path/to/player.yaml \
        --hole-spec path/to/hole.py \
        --conditions tournament \
        --baseline data/sg_optimizer/baselines/pga_tour.yaml

``--hole-spec`` is a Python file exporting a top-level ``HOLE`` SyntheticHole
object. (GeoJSON ingestion arrives in Phase 2.)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sg-optimizer")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--hole-spec", required=True)
    parser.add_argument("--conditions", default="tournament")
    parser.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="raster resolution in yards (default 2.0)",
    )
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--from-tee",
        action="store_true",
        help="report optimal action from the tee state",
    )
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
