"""Hand-to-hand relative-pose drift of the prescribed grip input (issue #11986).

Reproduce from the repository root (cheap, seconds)::

    PYTHONPATH=.:src python3 \\
        docs/development/full_body_models/evidence/grip_kinetics/run_grip_hand_drift.py

For each club, the prescribed hand grip frames of the contact run are built
exactly as ``ClubInHands.place_hands`` builds them (spline coordinates, weld
club forward kinematics, ``hand_frame_states``) at every 2 ms fixture sample.
The drift of the trail frame in the lead frame, relative to the first sample,
is written to ``contact/hand_drift_<club>.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python.grip_bushing import (  # noqa: E402
    WeldClubKinematics,
)
from src.shared.python.grip_contact import (  # noqa: E402
    CoordinateSpline,
    GripInterface,
    hand_frame_states,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.hand_drift import hand_relative_drift  # noqa: E402

MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
OUT = HERE / "contact"


def drift_for(club: str) -> dict:
    spec_bytes = (MODELS / f"full_body_spec_anthro_{club}.json").read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz",
        FIXTURES / "address_poses.json",
        club,
        names,
    )
    interface = GripInterface.from_spec(spec)
    kin = WeldClubKinematics(spec_bytes, names)
    spline = CoordinateSpline(swing.time_s, swing.q)
    rot: dict[str, list] = {"L": [], "R": []}
    pos: dict[str, list] = {"L": [], "R": []}
    for t in swing.time_s:
        hands = hand_frame_states(kin.state(*spline.evaluate(float(t))), interface)
        for s in "LR":
            rot[s].append(hands[s].rotation)
            pos[s].append(hands[s].position_m)
    pose = {s: (np.array(rot[s]), np.array(pos[s])) for s in "LR"}
    drift = hand_relative_drift(pose)
    sep = np.linalg.norm(pose["R"][1] - pose["L"][1], axis=1)
    return {
        "club": club,
        "samples": int(swing.time_s.size),
        "duration_s": float(swing.time_s[-1]),
        "hand_separation_mm": float(1e3 * sep[0]),
        "hand_separation_change_mm": float(1e3 * np.abs(sep - sep[0]).max()),
        "peak_drift": drift.peak,
        "note": "trail (R) frame in the lead (L) frame, change vs sample 0",
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for club in ("driver", "iron7"):
        info = drift_for(club)
        (OUT / f"hand_drift_{club}.json").write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8"
        )
        sys.stdout.write(json.dumps(info, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
