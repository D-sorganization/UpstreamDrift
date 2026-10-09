"""OpenSim contact-grip full-swing runs (issue #11739, OSV-7 phase 4, epic #11726).

Reproduce from the repository root, one heavy simulation at a time (the model
carries 24 ``ElasticFoundationForce`` pads on two closed grip meshes; the swing
takes hours, run it on the simulation host)::

    MPLBACKEND=Agg PYTHONPATH=.:src nice -n 10 python3 \\
        docs/development/full_body_models/evidence/grip_kinetics/run_grip_contact_opensim.py \\
        --club driver

The pad layout, friction, dissipation and per-hand squeeze are those of the
MuJoCo contact run (``run_grip_contact.py``): the squeeze is derived from the
OpenSim bushing demand of the same club, never tuned.  The elastic-foundation
stiffness is the analytic Winkler matching of the pad law, refined by one
static evaluation of the real mesh contact (``calibrated_foundation``).  The
summary JSON is comparable to ``contact/validation_<club>.json``: per-hand
peaks, net force, internal force, squeeze, couple and slip, next to the OpenSim
bushing of the same swing.  Outputs are overwritten by name; nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from run_grip_contact import squeeze_for  # noqa: E402

from src.shared.python.grip_contact import (  # noqa: E402
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model  # noqa: E402
from src.shared.python.grip_contact.parity import GripKineticsSeries  # noqa: E402

MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
OUT = HERE / "contact"
REFERENCE = HERE / "parity"
DEFAULT_ACCURACY = 1e-4


def peaks(series: GripKineticsSeries) -> dict[str, float]:
    """Per-hand peak force, net, internal, squeeze and couple of a series."""
    q = series.quantities()
    fl, fr = (np.linalg.norm(q[k], axis=1) for k in ("force_L", "force_R"))
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--club", choices=("driver", "iron7"), required=True)
    ap.add_argument("--accuracy", type=float, default=DEFAULT_ACCURACY)
    ap.add_argument("--method", default="CPodes")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from src.engines.physics_engines.opensim.python.grip_contact_osim_sim import (
        ContactGripSimulator,
    )

    spec_bytes = (MODELS / f"full_body_spec_anthro_{args.club}.json").read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{args.club}.npz",
        FIXTURES / "address_poses.json",
        args.club,
        names,
    )
    interface = GripInterface.from_spec(spec)
    squeeze = squeeze_for(args.club, interface)
    pads = build_pad_model(interface, squeeze)
    start = time.perf_counter()
    sim = ContactGripSimulator(
        spec_bytes, swing.names, swing.time_s, swing.q, pads, interface
    )
    build_s = time.perf_counter() - start
    sys.stderr.write(f"model built in {build_s:.1f} s\n")
    t0 = time.perf_counter()

    def progress(t: float) -> None:
        if round(t / 0.002) % 25 == 0:
            sys.stderr.write(f"t={t:.3f} s  wall={time.perf_counter() - t0:.1f} s\n")
            sys.stderr.flush()

    run = sim.run(
        t_end=args.t_end,
        accuracy=args.accuracy,
        on_sample=progress,
        method=args.method,
    )
    wall = time.perf_counter() - start
    ef = sim.contact_config.ef
    info: dict[str, Any] = {
        "club": args.club,
        "engine": "opensim_contact",
        "integrator": f"OpenSim Manager, {args.method}, error controlled",
        "accuracy": args.accuracy,
        "wall_time_s": wall,
        "build_time_s": build_s,
        "squeeze_per_hand_n": squeeze,
        "pad_stiffness_shared_n_m": pads.law.stiffness_n_m,
        "foundation_stiffness_n_m3": ef.stiffness_n_m3,
        "foundation_preload_penetration_m": ef.preload_penetration_m,
        "shared_preload_penetration_m": pads.layout.preload_penetration_m,
        "dissipation_s_m": ef.dissipation_s_m,
        "friction_static": ef.static_friction,
        "friction_dynamic": ef.dynamic_friction,
        "transition_velocity_m_s": ef.transition_velocity_m_s,
        "mesh_segments": sim.contact_config.mesh_segments,
        "ring_pitch_m": sim.contact_config.ring_pitch_m,
        "t_end_s": float(run.time_s[-1]),
    }
    contact = run.to_contact_run(info)
    OUT.mkdir(exist_ok=True)
    stem = f"opensim_{args.club}{args.tag}"
    contact.save_npz(OUT / f"{stem}_series.npz")
    reference = GripKineticsSeries.load_npz(
        REFERENCE / f"opensim_{args.club}_series.npz"
    )
    keep = reference.time_s <= run.time_s[-1] + 1e-9
    summary = {
        **info,
        "opensim_contact": peaks(contact.series),
        "opensim_bushing_same_window": peaks(
            GripKineticsSeries(
                engine="opensim",
                time_s=reference.time_s[keep],
                force_on_club_n={s: reference.force_on_club_n[s][keep] for s in "LR"},
                torque_on_club_nm={
                    s: reference.torque_on_club_nm[s][keep] for s in "LR"
                },
                grip_point_m={s: reference.grip_point_m[s][keep] for s in "LR"},
                deflection_m={s: reference.deflection_m[s][keep] for s in "LR"},
                rotation_deflection_rad={
                    s: reference.rotation_deflection_rad[s][keep] for s in "LR"
                },
                club_rotation=reference.club_rotation[keep],
            )
        ),
        "contact_slip": {
            "peak_roll_slip_rad": float(
                max(np.abs(contact.roll_slip_rad[s]).max() for s in "LR")
            ),
            "peak_axial_slip_mm": float(
                max(np.abs(contact.axial_slip_m[s]).max() for s in "LR") * 1e3
            ),
        },
        "pad_normal_force_sum_peak_n": float(
            max(contact.squeeze_n[s].max() for s in "LR")
        ),
        "hand_to_club_displacement_peak_mm": float(
            max(
                np.linalg.norm(contact.series.deflection_m[s], axis=1).max()
                for s in "LR"
            )
            * 1e3
        ),
        "hand_to_club_rotation_peak_deg": float(
            np.degrees(
                max(
                    np.abs(contact.series.rotation_deflection_rad[s]).max()
                    for s in "LR"
                )
            )
        ),
    }
    (OUT / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
