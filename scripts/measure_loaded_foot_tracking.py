#!/usr/bin/env python3
"""Measure the loaded squat's foot-joint tracking against RK4 substep and hold.

OSL-16 (#13104).  Rebuilds the bar-balanced IK of the committed back-squat
bundle from its provenance, tracks its window in MuJoCo on the bundle's
rigid-bar spec with the pipeline controller for each (RK4 substeps, effort
hold) policy of the grid, and records per run:

* ``foot_joint_error_rad``: largest |q - q_IK| of ankle, subtalar and mtp;
* ``foot_point_deviation_m``: largest tibia-frame foot-point distance from
  the IK per side (:mod:`src.shared.python.lifting.loaded_foot`), the length
  bounded by the side's LIFT-5 ankle IK residual;
* ``rk4_stability``: the stiffest open-loop rate on the run's own states
  times its substep (``max_abs_z``; RK4 is stable on the real axis below
  2.785);
* the run's ``failure`` and reached step count.

The record is written as JSON; ``tests/unit/lifting/test_loaded_foot_tracking.py``
reads it.  Runs in parallel (``--jobs``) and takes about 5 minutes per run on
one core.  Usage::

    python3 scripts/measure_loaded_foot_tracking.py --jobs 4
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.lifting.bundles import (  # noqa: E402
    BACK_SQUAT_BUNDLE,
    LIFT_SPEC,
    lift_to_full_body,
    load_seed_pose,
)
from src.shared.python.motion_matching.same_input import (  # noqa: E402
    InputBundle,
    StepPolicy,
)

logger = logging.getLogger(__name__)
RECORD = BACK_SQUAT_BUNDLE.with_name("loaded_foot_tracking.json")
#: (RK4 substeps per 1 ms step, effort hold in s).  The first is the original
#: contract; the substep column at a 1 ms hold isolates RK4 stability, the
#: 32-substep row isolates the hold.
GRID: tuple[tuple[int, float], ...] = (
    (8, 1e-3),
    (8, 1.25e-4),
    (16, 1e-3),
    (32, 1e-3),
    (32, 5e-4),
    (64, 5e-4),
    (32, 2.5e-4),
    (32, 1.25e-4),
)
STABILITY_EVERY = 20


def bundle_window(bundle: InputBundle) -> tuple[np.ndarray, np.ndarray, float]:
    """Times and IK joint trajectory of ``bundle``'s window, rebuilt from its
    provenance (bar-balanced IK on the lift body), and the window length."""
    from src.shared.python.lifting.back_bar import BackBarGeometry
    from src.shared.python.lifting.loaded_balance import load_balance_from_record
    from src.shared.python.lifting.sagittal_retarget import SagittalSolution

    prov = bundle.provenance
    balance = prov["ik"].get("balance")
    if balance is None:
        raise ValueError("bundle is not bar-load balanced (#13093)")
    geom = prov["ik"]["back_bar"]["geometry"]
    solution = SagittalSolution.from_record(prov["sagittal_solution"])
    lift = lift_to_full_body(
        LIFT_SPEC.read_bytes(),
        solution,
        load_seed_pose(),
        geometry=BackBarGeometry.from_standard(geom["lift"], height_m=geom["height_m"]),
        load=load_balance_from_record(balance["load"]),
    )
    start = int(np.searchsorted(solution.time_s, prov["window"]["start_s"] - 1e-12))
    return solution.time_s[start:], lift.q[start:], float(prov["window"]["duration_s"])


def measure(args: tuple[bytes, list[str], np.ndarray, np.ndarray, float, int, float]):
    """One grid run: track, then measure the foot against the IK."""
    from src.shared.python.lifting.loaded_foot import (
        foot_joint_error_rad,
        foot_point_deviation_m,
    )
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        closed_loop,
        hold_policy,
    )
    from src.shared.python.motion_matching.same_input.stiffness import (
        rk4_stability_record,
    )

    spec, order, times, track, window, substeps, hold = args
    policy = hold_policy(hold, substeps)
    started = time.monotonic()
    rollout = closed_loop(
        "mujoco",
        spec,
        times,
        track,
        duration_s=window,
        policy=StepPolicy(
            substeps=policy.substeps,
            effort_holds=policy.effort_holds,
            stop_on_failure=True,
        ),
    )
    clock = times - times[0]
    ref = np.column_stack(
        [np.interp(rollout.time_s, clock, track[:, k]) for k in range(track.shape[1])]
    )
    rows = np.arange(0, rollout.efforts.shape[0], STABILITY_EVERY * policy.effort_holds)
    steps = rows // policy.effort_holds
    return {
        "substeps": substeps,
        "effort_hold_s": hold,
        "failure": rollout.failure,
        "reached_steps": int(rollout.q.shape[0] - 1),
        "foot_joint_error_rad": foot_joint_error_rad(order, rollout.q, ref),
        "foot_point_deviation_m": foot_point_deviation_m(spec, order, rollout.q, ref),
        "rk4_stability": rk4_stability_record(
            VectorPlant("mujoco", spec),
            rollout.q[steps],
            rollout.v[steps],
            rollout.efforts[rows],
            1e-3 / substeps,
        ),
        "wall_s": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=BACK_SQUAT_BUNDLE)
    parser.add_argument("--out", type=Path, default=RECORD)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    bundle = InputBundle.load(args.bundle)
    times, track, window = bundle_window(bundle)
    shared = (bytes(bundle.spec_bytes), list(bundle.coordinate_order), times, track)
    jobs = [(*shared, window, substeps, hold) for substeps, hold in GRID]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        runs = list(pool.map(measure, jobs))
    for run in runs:
        logger.info(
            "substeps %3d hold %.3e s: failure %s, |z| %.2f, foot r/l %.2f/%.2f mm",
            run["substeps"],
            run["effort_hold_s"],
            run["failure"],
            run["rk4_stability"]["max_abs_z"],
            1e3 * run["foot_point_deviation_m"]["r"],
            1e3 * run["foot_point_deviation_m"]["l"],
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    record: dict[str, Any] = {
        "issue": 13104,
        "generator": "scripts/measure_loaded_foot_tracking.py",
        "source_commit": commit,
        "bundle": str(args.bundle.relative_to(ROOT)),
        "bundle_spec_sha256": bundle.spec_sha256,
        "bundle_policy": bundle.manifest()["policy"],
        "window": bundle.provenance["window"],
        "runs": runs,
    }
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    logger.info("wrote %s", args.out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
