"""Annotate a saved matched-swing run with finish-feasibility metrics (#11668).

Reads ``dynamics_record.npz`` (simulation record and the tracked reference) and
the run's scaled full-body specification, evaluates the finish-feasibility
block and prints it, or writes it into the run's ``receipt.json`` under
``dynamics.finish_feasibility``.

    python -m src.shared.python.motion_matching.pipeline.finish_feasibility_cli \\
        --run-dir <run> [--receipt <run>/receipt.json] [--write]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.shared.python.motion_matching.pipeline.finish_feasibility import (
    finish_feasibility_report,
)

logger = logging.getLogger(__name__)

SPEC_NAME = "full_body_spec_hipcal_scaled.json"
RECORD_NAME = "dynamics_record.npz"
RECORD_KEYS = (
    "time_s",
    "q",
    "v",
    "weight_fraction",
    "inside",
    "q_track",
    "track_time_s",
)


def load_record(run_dir: Path) -> dict[str, np.ndarray]:
    """Load the arrays of ``dynamics_record.npz`` the metrics need."""
    path = run_dir / RECORD_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in RECORD_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"{path} lacks arrays {missing}")
        return {k: np.asarray(data[k]) for k in RECORD_KEYS}


def compute_for_run(run_dir: Path) -> dict[str, Any]:
    """Finish-feasibility block of a saved run directory."""
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    spec = run_dir / SPEC_NAME
    if not spec.is_file():
        raise FileNotFoundError(f"missing {spec}")
    rec = load_record(run_dir)
    sim = fs.FullBodySimulator(NativeMujocoFullBodyModel(spec.read_bytes()))
    ground = sim.adapter.ground_plane
    zmp = fs.reference_zmp(sim, rec["track_time_s"], rec["q_track"], ground)
    record = fs.SimulationRecord(
        time_s=rec["time_s"],
        q=rec["q"],
        v=rec["v"],
        tau=np.zeros_like(rec["q"]),
        normal_force_n=np.zeros(len(rec["time_s"])),
        weight_fraction=rec["weight_fraction"],
        centre_of_pressure_m=np.zeros((len(rec["time_s"]), 3)),
        inside_support_polygon=rec["inside"].astype(bool),
        lowest_sphere_height_m=np.zeros(len(rec["time_s"])),
    )
    return finish_feasibility_report(
        sim,
        times_track=rec["track_time_s"],
        q_track=rec["q_track"],
        record=record,
        zmp=zmp,
        ground=ground,
    )


def annotate_receipt(receipt_path: Path, block: dict[str, Any]) -> None:
    """Write ``block`` into ``dynamics.finish_feasibility`` of the receipt."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if "dynamics" not in receipt:
        raise ValueError(f"{receipt_path} has no dynamics block")
    receipt["dynamics"]["finish_feasibility"] = block
    receipt_path.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the post-processing CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--write", action="store_true", help="annotate the receipt in place"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; the JSON block goes to stdout (wire output)."""
    args = build_parser().parse_args(argv)
    block = compute_for_run(args.run_dir)
    if args.write:
        receipt = args.receipt or args.run_dir / "receipt.json"
        annotate_receipt(receipt, block)
        logger.info("annotated %s", receipt)
    else:
        sys.stdout.write(json.dumps(block, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
