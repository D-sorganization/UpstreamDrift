"""Same-input bushing-grip parity across OpenSim, MuJoCo, Drake and Pinocchio.

Issue #11739 (OSV-7 phase 2), epic #11726.  Reproduce from the repository
root, one heavy simulation at a time::

    MPLBACKEND=Agg PYTHONPATH=.:src python3 \\
        docs/development/full_body_models/evidence/grip_kinetics/run_grip_parity.py \\
        --club driver --reference          # OpenSim reference series (slow)
    ... run_grip_parity.py --club driver --engines mujoco drake pinocchio
    ... run_grip_parity.py --club driver --report   # metrics json and plots

Input: the OSV-10 fixture ``tests/fixtures/club_face/swing_q_<club>.npz``
mapped by name onto ``full_body_spec_anthro_<club>.json``.  Every engine
prescribes the same coordinate spline (``grip_contact.CoordinateSpline``,
identical to OpenSim's ``SimmSpline``), computes the hand frames from its own
forward kinematics of the weld model and integrates a free club held by two
bushings.  Outputs are overwritten by name; nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from src.shared.python.grip_contact import load_coordinate_swing  # noqa: E402
from src.shared.python.grip_contact.parity import (  # noqa: E402
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    GripKineticsSeries,
    parity_errors,
)

MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
OUT = HERE / "parity"
PLOTS = Path(
    "/home/dieterolson/Videos/Parity Audit/golfer_realism/grip_kinetics/parity"
)
ENGINE_MODULES = {
    "mujoco": "src.engines.physics_engines.mujoco.python.grip_bushing",
    "drake": "src.engines.physics_engines.drake.python.grip_bushing",
    "pinocchio": "src.engines.physics_engines.pinocchio.python.grip_bushing",
}
REFERENCE_ACCURACY = 1e-5


def _spec_bytes(club: str) -> bytes:
    return (MODELS / f"full_body_spec_anthro_{club}.json").read_bytes()


def _swing(club: str) -> Any:
    names = json.loads(_spec_bytes(club))["coordinate_order"]
    return load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz", FIXTURES / "address_poses.json", club, names
    )


def _series_path(engine: str, club: str) -> Path:
    return OUT / f"{engine}_{club}_series.npz"


def run_reference(club: str, accuracy: float) -> dict[str, Any]:
    """OpenSim BushingForce reference (RK-Merson, error controlled)."""
    from src.engines.physics_engines.opensim.python.grip_bushing_sim import (
        BushingGripSimulator,
    )

    swing = _swing(club)
    start = time.perf_counter()
    sim = BushingGripSimulator(_spec_bytes(club), swing.names, swing.time_s, swing.q)
    run = sim.run(accuracy=accuracy)
    wall = time.perf_counter() - start
    series = GripKineticsSeries.from_bushing_run("opensim", run)
    series.save_npz(_series_path("opensim", club))
    return {"engine": "opensim", "accuracy": accuracy, "wall_time_s": wall}


def run_engine(engine: str, club: str) -> dict[str, Any]:
    """Run one engine on the same input and save its series."""
    module = import_module(ENGINE_MODULES[engine])
    start = time.perf_counter()
    series = module.simulate_grip_bushing(_spec_bytes(club), _swing(club))
    wall = time.perf_counter() - start
    series.save_npz(_series_path(engine, club))
    return {"engine": engine, "wall_time_s": wall, **dict(series.metadata)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club", choices=("driver", "iron7"), required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--accuracy", type=float, default=REFERENCE_ACCURACY)
    parser.add_argument("--engines", nargs="*", default=[])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"runs_{args.club}.json"
    runs = json.loads(log.read_text(encoding="utf-8")) if log.is_file() else {}
    if args.reference:
        runs["opensim"] = run_reference(args.club, args.accuracy)
    for engine in args.engines:
        runs[engine] = run_engine(engine, args.club)
    log.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
