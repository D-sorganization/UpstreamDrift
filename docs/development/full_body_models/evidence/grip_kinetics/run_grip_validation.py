"""Kinetics validation of the contact grip (issue #11739, OSV-7 phase 3).

Reproduce from the repository root after ``run_grip_contact.py`` for the club::

    MPLBACKEND=Agg PYTHONPATH=.:src nice -n 10 python3 \\
        docs/development/full_body_models/evidence/grip_kinetics/run_grip_validation.py \\
        --club driver

Three reports are written to ``contact/validation_<club>.json``:

* ``quasi_static``: the MuJoCo contact grip holds the club still at several
  swing attitudes; the force and moment balance of the extracted per-hand
  wrenches must close (``grip_contact.static_balance``).
* ``indeterminacy``: the left/right split and the internal force pair of the
  weld (minimum-norm proxy), the OpenSim bushing and the contact grip.  The
  two hands holding one rigid club are statically indeterminate; each grip
  model resolves the split differently and none is data.
* ``deflection``: the bushing deflection at peak load against the plan's
  3 mm / 2 degree flag (flag only; never tuned) and the contact slip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from src.shared.python.biomechanics.grip_wrench import allocate_min_norm  # noqa: E402
from src.shared.python.grip_contact import (  # noqa: E402
    ClubDynamics,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.contact_run import ContactRun  # noqa: E402
from src.shared.python.grip_contact.pad_contact import build_pad_model  # noqa: E402
from src.shared.python.grip_contact.parity import GripKineticsSeries  # noqa: E402
from src.shared.python.grip_contact.static_balance import static_balance  # noqa: E402

MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
OUT = HERE / "contact"
PARITY = HERE / "parity"
DEFLECTION_FLAG_M = 3.0e-3
ROTATION_FLAG_DEG = 2.0
HOLD_S = 0.2
POSE_FRACTIONS = (0.0, 0.4, 0.55, 0.7)


def quasi_static(club: str, squeeze: float) -> dict[str, Any]:
    from src.engines.physics_engines.mujoco.python import grip_contact_sim as sim_mod

    spec_bytes = (MODELS / f"full_body_spec_anthro_{club}.json").read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz", FIXTURES / "address_poses.json", club, names
    )
    interface = GripInterface.from_spec(spec)
    dyn = ClubDynamics.from_spec(spec)
    g = np.asarray(spec["gravity_m_s2"], float)
    sim = sim_mod.ClubInHands(
        spec_bytes, names, interface, build_pad_model(interface, squeeze)
    )
    q0 = np.asarray(swing.q[0], float)
    sim.calibrate(q0)
    poses = []
    for frac in POSE_FRACTIONS:
        k = int(round(frac * (len(swing.q) - 1)))
        run = sim_mod.hold_run(sim, np.asarray(swing.q[k], float), HOLD_S)
        bal = static_balance(run.series, interface, dyn, g)
        poses.append(  # first sample is the release instant, not equilibrium
            {
                "swing_time_s": float(swing.time_s[k]),
                "force_residual_fraction_incl_release_sample": bal.max_force_error(),
                "moment_residual_fraction_incl_release_sample": bal.max_moment_error(),
                "gravity_moment_nm": float(bal.gravity_moment_nm[-1]),
                "settled_force_residual_fraction": float(
                    np.linalg.norm(bal.force_residual_n[-1]) / bal.weight_n
                ),
                "settled_moment_residual_fraction": float(
                    np.linalg.norm(bal.moment_residual_nm[-1])
                    / max(bal.gravity_moment_nm[-1], 1e-12)
                ),
                "axial_slip_mm": float(
                    max(np.abs(run.axial_slip_m[s]).max() for s in "LR") * 1e3
                ),
            }
        )
    return {"hold_s": HOLD_S, "poses": poses}


def _split(series: GripKineticsSeries) -> dict[str, float]:
    q = series.quantities()
    fl = np.linalg.norm(q["force_L"], axis=1)
    fr = np.linalg.norm(q["force_R"], axis=1)
    k = int(np.argmax(np.linalg.norm(q["net_force"], axis=1)))
    return {
        "peak_force_L_n": float(fl.max()),
        "peak_force_R_n": float(fr.max()),
        "lead_share_at_peak_net": float(fl[k] / (fl[k] + fr[k])),
        "peak_net_n": float(np.linalg.norm(q["net_force"], axis=1).max()),
        "peak_internal_n": float(np.linalg.norm(q["internal_force"], axis=1).max()),
        "peak_squeeze_n": float(np.abs(q["squeeze"]).max()),
        "peak_couple_nm": float(np.linalg.norm(q["couple"], axis=1).max()),
    }


def weld_proxy(series: GripKineticsSeries) -> dict[str, float]:
    """Minimum-norm split of the same net wrench (a labelled proxy, not a model)."""
    left, right, net = [], [], []
    for a in series.analyses():
        fl, fr = allocate_min_norm(a)
        left.append(np.linalg.norm(fl))
        right.append(np.linalg.norm(fr))
        net.append(np.linalg.norm(a.net_force_n))
    left, right, net = map(np.asarray, (left, right, net))
    k = int(np.argmax(net))
    return {
        "peak_force_L_n": float(left.max()),
        "peak_force_R_n": float(right.max()),
        "lead_share_at_peak_net": float(left[k] / (left[k] + right[k])),
        "peak_net_n": float(net.max()),
    }


def deflection(series: GripKineticsSeries) -> dict[str, float]:
    mm = max(float(np.linalg.norm(series.deflection_m[s], axis=1).max()) for s in "LR")
    deg = max(float(np.abs(series.rotation_deflection_rad[s]).max()) for s in "LR")
    deg = float(np.degrees(deg))
    return {
        "max_translation_mm": mm * 1e3,
        "max_rotation_deg": deg,
        "flag_translation_mm": DEFLECTION_FLAG_M * 1e3,
        "flag_rotation_deg": ROTATION_FLAG_DEG,
        "exceeds_flag": bool(mm > DEFLECTION_FLAG_M or deg > ROTATION_FLAG_DEG),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--club", choices=("driver", "iron7"), required=True)
    ap.add_argument("--skip-hold", action="store_true")
    args = ap.parse_args()
    club = args.club
    info = json.loads((OUT / f"mujoco_{club}_run.json").read_text())
    ref = GripKineticsSeries.load_npz(PARITY / f"opensim_{club}_series.npz")
    contact = ContactRun.load_npz(OUT / f"mujoco_{club}_series.npz")
    report: dict[str, Any] = {"club": club}
    if not args.skip_hold:
        report["quasi_static"] = quasi_static(club, info["squeeze_per_hand_n"])
    report["indeterminacy"] = {
        "weld_min_norm_proxy": weld_proxy(ref),
        "opensim_bushing": _split(ref),
        "mujoco_contact": _split(contact.series),
        "note": "two hands on one rigid club are statically indeterminate; "
        "the weld row is a minimum-norm proxy of the opensim net wrench",
    }
    report["deflection"] = {
        "opensim_bushing": deflection(ref),
        "mujoco_contact_hand_to_club": deflection(contact.series),
        "contact_slip": {
            "peak_roll_slip_rad": float(
                max(np.abs(contact.roll_slip_rad[s]).max() for s in "LR")
            ),
            "peak_axial_slip_mm": float(
                max(np.abs(contact.axial_slip_m[s]).max() for s in "LR") * 1e3
            ),
        },
    }
    path = OUT / f"validation_{club}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
