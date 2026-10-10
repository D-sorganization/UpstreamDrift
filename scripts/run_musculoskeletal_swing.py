"""Run the OpenSim musculoskeletal golf-swing pipeline end to end (issue #11617).

    MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen PYTHONPATH=.:src \
        python3 scripts/run_musculoskeletal_swing.py --out-dir /tmp/msk_run \
        --receipt docs/development/full_body_models/evidence/musculoskeletal/receipt.json

Builds the Rajagopal-Lai-Uhlrich muscle model fitted to the golf humanoid, maps
the matched swing IK states onto it, estimates ground reactions, solves
StaticOptimization and writes a receipt.  Headless; never launches a GUI.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.engines.physics_engines.opensim.python.musculoskeletal_pipeline import (
    PipelineConfig,
    run_pipeline,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_solvers import (
    SolveWindow,
)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/development/opensim_tour_matching/evidence/os7_moco_g1/inputs"
DEFAULT_GOLF = EVIDENCE / "golf_humanoid_scaled_tour_markers.osim"
DEFAULT_STATES = EVIDENCE / "ik_states_full.sto"
DEFAULT_RECEIPT = (
    REPO / "docs/development/full_body_models/evidence/musculoskeletal/receipt.json"
)

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golf-model", type=Path, default=DEFAULT_GOLF)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=1.45)
    parser.add_argument("--cutoff-hz", type=float, default=15.0)
    parser.add_argument("--so-step", type=int, default=2)
    parser.add_argument(
        "--moco-pilot",
        type=float,
        nargs=3,
        metavar=("T0", "T1", "MESH_S"),
        help="also run a short MocoInverse pilot on [T0, T1] with this mesh interval",
    )
    parser.add_argument("--moco-iterations", type=int, default=25)
    parser.add_argument("--phase-split", type=float, default=1.08)
    parser.add_argument(
        "--club",
        choices=("driver", "iron7", "none"),
        default="driver",
        help="shared club held in both hands; 'none' builds the no-club model",
    )
    parser.add_argument(
        "--club-control",
        action="store_true",
        help="also solve the no-club control and record the club's upper-body load",
    )
    parser.add_argument(
        "--enforce-trail-weld",
        action="store_true",
        help="keep the trail-hand weld (the IK does not close the two-hand loop)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = PipelineConfig(
        golf_model=args.golf_model,
        states_file=args.states,
        out_dir=args.out_dir,
        window=SolveWindow(args.t_start, args.t_end),
        cutoff_hz=args.cutoff_hz,
        so_step=args.so_step,
        phase_split_s=args.phase_split,
        base_model=args.base_model,
        club=None if args.club == "none" else args.club,
        enforce_trail_weld=args.enforce_trail_weld,
    )
    receipt = run_pipeline(
        cfg,
        args.receipt,
        moco_pilot=args.moco_pilot,
        moco_iterations=args.moco_iterations,
        club_control=args.club_control,
    )
    logger.info("receipt: %s", args.receipt)
    logger.info("%s", json.dumps(receipt["results"]["per_group"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
