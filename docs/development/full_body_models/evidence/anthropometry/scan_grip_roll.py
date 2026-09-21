"""Calibrate the hand roll about the shaft from the swing (MM-2, #10104).

For each grip roll the driver document is rebuilt, the capture is matched
with the wrists unbounded (flagged only) and the wrist and forearm
coordinate excursions beyond the human ranges are summed over the
full-capture IK. The roll with the smallest total excursion is the one under
which the model's wrist axes are closest to the golfer's; it is then the
value to impose the ranges with. Writes ``scan_grip_roll_receipt.json``.

    python scan_grip_roll.py -60 -30 0 30 60
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.shared.python.motion_matching.range_of_motion import (  # noqa: E402
    HUMAN_RANGES_DEG,
    violations,
)

HERE = Path(__file__).resolve().parent
FULL_BODY = ROOT / "docs/development/full_body_models"
BUILD = FULL_BODY / "build_anthropometric_spec.py"
DRIVER = FULL_BODY / "evidence/ground_support/run_ground_support.py"
NATIVE = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
OSIM = ROOT / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
CANDIDATE = FULL_BODY / "evidence/native_candidates/returned81_candidate.json"
WRISTS = ("LWInputX", "LWInputY", "LFInput", "RWInputX", "RWInputY", "RFInput")


def evaluate(roll: float, capture: str, club: str) -> dict:
    name = f"scan_roll_{club}_{roll:+.0f}"
    out = HERE / "scan" / name
    subprocess.run(
        [
            sys.executable, str(BUILD), "--native", str(NATIVE), "--osim", str(OSIM),
            "--native-candidate", str(CANDIDATE), "--stature", "1.71", "--mass", "78",
            "--trunk-scale", "1.15", "--arm-scale", "1.1", "--shoulder-scale", "1.0",
            "--club", club, "--grip-roll", str(roll), "--output", str(HERE / "scan"),
            "--name", name,
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    subprocess.run(
        [
            sys.executable, str(DRIVER), "--spec", str(HERE / "scan" / f"{name}.json"),
            "--skip-hip-calibration", "--static-seeds", "--capture", capture,
            "--out", str(out),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    receipt = json.loads((out / "receipt.json").read_text())
    q = np.load(out / "ik_trajectory.npz")["q"]
    names = json.loads((out / "full_body_spec_hipcal_scaled.json").read_text())[
        "coordinate_order"
    ]
    found = violations(q, names, HUMAN_RANGES_DEG)
    deg = np.degrees(q)
    return {
        "grip_roll_deg": roll,
        "ik_rms_mm": round(receipt["ik"]["marker_rms_m"] * 1e3, 1),
        "wrist_excess_deg": {
            n: round(found[n].max_excess_deg, 1) for n in WRISTS if n in found
        },
        "total_wrist_excess_deg": round(
            sum(found[n].max_excess_deg for n in WRISTS if n in found), 1
        ),
        "address_deg": {n: round(float(deg[0, names.index(n)]), 1) for n in WRISTS},
        "range_deg": {
            n: [
                round(float(deg[:, names.index(n)].min()), 1),
                round(float(deg[:, names.index(n)].max()), 1),
            ]
            for n in WRISTS
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("grip_roll")
    rolls = [float(v) for v in sys.argv[1:]] or [-60.0, -30.0, 0.0, 30.0, 60.0]
    capture, club = "driver", "driver"
    receipt_path = HERE / "scan_grip_roll_receipt.json"
    rows = json.loads(receipt_path.read_text())["rows"] if receipt_path.exists() else []
    for roll in rolls:
        row = evaluate(roll, capture, club)
        rows = [r for r in rows if r["grip_roll_deg"] != roll] + [row]
        log.info(
            "roll %+.0f: IK %.1f mm, wrist excess %s (total %.1f)",
            roll, row["ik_rms_mm"], row["wrist_excess_deg"], row["total_wrist_excess_deg"],
        )  # fmt: skip
        receipt_path.write_text(
            json.dumps(
                {
                    "capture": capture,
                    "club": club,
                    "metric": "sum of wrist and forearm excursions beyond HUMAN_RANGES_DEG over the full-capture IK, wrists unbounded",
                    "rows": sorted(rows, key=lambda r: r["total_wrist_excess_deg"]),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
