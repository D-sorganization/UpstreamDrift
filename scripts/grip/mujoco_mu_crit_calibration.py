"""MuJoCo mu_crit hold calibration receipt (#13885): ``python3 -m scripts.grip.mujoco_mu_crit_calibration``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from src.engines.physics_engines.mujoco.python.grip_hold_calibration import (
    BIAS_BAND,
    BISECTION_TOL,
    PAD_SOLREF,
    RECEIPT_INTEGRATORS,
    HoldRig,
    calibrate,
)
from src.shared.python.grip_contact.grip_margin import DEFAULT_SLIP_SPEED_M_S

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/grip_kinetics"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrators", nargs="+", default=list(RECEIPT_INTEGRATORS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    rig = HoldRig()
    rows = [asdict(calibrate(rig, integrator)) for integrator in args.integrators]
    receipt = {
        "issue": 13885,
        "rig": asdict(rig),
        "pad_solref": PAD_SOLREF,
        "slip_speed_m_s": DEFAULT_SLIP_SPEED_M_S,
        "bias_band": BIAS_BAND,
        "bisection_tol_ratio": BISECTION_TOL,
        "calibrations": rows,
        "reproduce": "python3 -m scripts.grip.mujoco_mu_crit_calibration",
    }
    out = args.out or EVIDENCE / "mujoco_mu_crit_calibration.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
